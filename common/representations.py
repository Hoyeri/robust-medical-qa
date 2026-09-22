"""Capture attention contributions and differentiate diagnostic readouts."""
from .io import require
from .runtime import ModelShape, forward
from .attention import receivers


class Capture:
    def __init__(self, model):
        self.model = model
        self.shape = ModelShape.from_model(model)
        self.value, self.weights, self.oproj, self.add = {}, {}, {}, {}
        self.handles = []

    def __enter__(self):
        import torch
        from transformers.models.llama import modeling_llama as impl
        self.impl, self.original = impl, impl.eager_attention_forward
        require(not getattr(self.original, '_medical_qa_wrapper', False), 'Nested attention wrapper')
        modules = {id(block.self_attn) for block in self.model.model.layers}
        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            result = self.original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if id(module) in modules:
                self.value[module.layer_idx] = value.detach()
                self.weights[module.layer_idx] = result[1].detach()
            return result
        wrapped._medical_qa_wrapper = True
        impl.eager_attention_forward = wrapped
        try:
            for index, block in enumerate(self.model.model.layers):
                def hook(module, args, output, index=index):
                    if index in self.add:
                        rows, extra = self.add[index]
                        output = output.clone()
                        output[0, rows, :] += extra.to(output.dtype)
                    if output.requires_grad:
                        output.retain_grad()
                    self.oproj[index] = output
                    return output
                self.handles.append(block.self_attn.o_proj.register_forward_hook(hook))
            self.handles.append(self.model.model.embed_tokens.register_forward_hook(
                lambda m, i, out: out.requires_grad_(True) if torch.is_grad_enabled() else None))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        self.impl.eager_attention_forward = self.original
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def clear(self):
        self.value.clear()
        self.weights.clear()
        self.oproj.clear()
        self.add.clear()

    def contribution(self, layer, source, query):
        import torch
        a = self.weights[layer][0][:, query][:, :, source].float()
        v = self.value[layer][0][:, source, :].float().repeat_interleave(self.shape.heads // self.shape.kv_heads, dim=0)
        c = torch.einsum('hqs,hsd->qhd', a, v).reshape(len(query), self.shape.heads * self.shape.head_dim)
        weight = self.model.model.layers[layer].self_attn.o_proj.weight.detach().float()
        return c, c @ weight.T


def hidden_features(model, ids, phrase, pooling_dtype='native'):
    import torch
    require(pooling_dtype in ('native', 'float32'), 'Unknown pooling dtype')
    with torch.no_grad():
        hidden = torch.stack(forward(model, ids, hidden=True).hidden_states)[:, 0]
    phrase_hidden = hidden[:, phrase, :]
    if pooling_dtype == 'float32':
        phrase_hidden = phrase_hidden.float()
    return {'phrase': phrase_hidden.mean(1).float().cpu().numpy(),
            'answer': hidden[:, -1, :].float().cpu().numpy()}


def transmission_features(model, ids, phrase, sources, label_ids, gold, target,
                          head_layers, validation_layers=(), alphas=()):
    import torch
    import numpy as np
    shape = ModelShape.from_model(model)
    require(all(0 <= l < shape.layers for l in list(head_layers) + list(validation_layers)), 'Requested layer outside model')
    requires = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with Capture(model) as capture:
            with torch.enable_grad():
                output = forward(model, ids, hidden=True)
                logits = output.logits[0, -1].float()
                q = logits[label_ids[target]] - logits[label_ids[gold]]
                q.backward()
            q_value = float(q.detach())
            gradients = {l: capture.oproj[l].grad[0].float() for l in range(shape.layers)}
            record = {'q': q_value, 'rs_label_logits': {k: float(logits[v].detach()) for k, v in label_ids.items()}, 'n_tokens': len(ids), 'sources': {}}
            record['argmax'] = max(record['rs_label_logits'], key=record['rs_label_logits'].get)
            for name, src in sources.items():
                query = receivers('all_source_cut', len(ids), src)
                require(bool(query) and query[-1] == len(ids)-1, 'Answer receiver missing')
                all_values, answer_values, all_heads, answer_heads = [], [], {}, {}
                for layer in range(shape.layers):
                    c, projected = capture.contribution(layer, src, query)
                    g = gradients[layer][query]
                    all_values.append(float((g * projected).sum()))
                    answer_values.append(float((g[-1] * projected[-1]).sum()))
                    if layer in head_layers:
                        weight = model.model.layers[layer].self_attn.o_proj.weight.detach().float()
                        per_head = ((g @ weight) * c).reshape(len(query), shape.heads, shape.head_dim).sum(-1)
                        all_heads[str(layer)], answer_heads[str(layer)] = per_head.sum(0).tolist(), per_head[-1].tolist()
                record['sources'][name] = dict(n_span_tokens=len(src), A_all=all_values, A_ans=answer_values,
                                               A_all_head=all_heads, A_ans_head=answer_heads)
            features = {
                'h': torch.stack([h[0, phrase, :].detach().float().mean(0) for h in output.hidden_states]).cpu().numpy(),
                'v': torch.stack([capture.value[l][0][:, phrase, :].float().mean(1).reshape(-1) for l in range(shape.layers)]).cpu().numpy()}
            c_values = [capture.contribution(l, phrase, [len(ids)-1]) for l in range(shape.layers)]
            features['av'] = torch.stack([c[0] for c, _ in c_values]).cpu().numpy()
            features['oav'] = torch.stack([p[0] for _, p in c_values]).cpu().numpy()
            # Freeze all validation contributions before any intervention forward updates the capture.
            query = receivers('all_source_cut', len(ids), sources['sent'])
            baseline_contributions = {l: capture.contribution(l, sources['sent'], query)[1].detach().clone() for l in validation_layers}
            validation = []
            for layer, contribution in baseline_contributions.items():
                for alpha in alphas:
                    capture.add = {layer: (query, alpha * contribution)}
                    with torch.no_grad():
                        changed = forward(model, ids).logits[0, -1].float()
                    capture.add.clear()
                    observed = float(changed[label_ids[target]] - changed[label_ids[gold]]) - q_value
                    validation.append(dict(layer=layer, alpha=alpha, dq_actual=observed,
                        dq_first_order=alpha * record['sources']['sent']['A_all'][layer]))
            require(all(np.isfinite(v).all() for v in features.values()), 'Non-finite extracted features')
            return features, record, validation
    finally:
        for p, enabled in zip(model.parameters(), requires):
            p.requires_grad_(enabled)

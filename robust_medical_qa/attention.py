"""Source/receiver attention blocking for eager Llama attention."""
from contextlib import contextmanager
from .io import require
from .runtime import ModelShape


def positions_ok(values, length):
    require(len(values) == len(set(values)) and all(type(x) is int and 0 <= x < length for x in values), 'Invalid token positions')


def receiver_mask(mask, queries, sources):
    import torch
    require(mask is not None and mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[-1] == mask.shape[-2], 'Expected full-prefix batch-one mask')
    length = mask.shape[-1]
    positions_ok(queries, length)
    positions_ok(sources, length)
    require(not set(queries) & set(sources), 'Source and receivers overlap')
    require(0 not in sources and len(sources) < length, 'Keep causal key zero available')
    if not queries or not sources:
        return mask
    q, s = torch.tensor(queries, device=mask.device), torch.tensor(sources, device=mask.device)
    result = mask.clone()
    result[:, :, q[:, None], s[None, :]] = float('-inf')
    return result


def receivers(policy, length, source, options=()):
    require(policy in ('baseline', 'option_cut', 'all_source_cut'), 'Unknown receiver policy')
    positions_ok(source, length)
    positions_ok(options, length)
    require(bool(source) and 0 not in source and not set(source) & set(options), 'Invalid source/options')
    if policy == 'baseline':
        return []
    if policy == 'option_cut':
        return list(options)
    return [p for p in range(min(source) + 1, length) if p not in set(source)]


@contextmanager
def receiver_edges(model, layers, queries, sources, verify=True):
    import torch
    from transformers.models.llama import modeling_llama as impl
    shape = ModelShape.from_model(model)
    positions_ok(layers, shape.layers)
    original = impl.eager_attention_forward
    require(not getattr(original, '_medical_qa_wrapper', False), 'Nested attention wrapper')
    modules = {id(block.self_attn) for block in model.model.layers}
    calls = []
    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if id(module) not in modules:
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        require(query.shape[0] == 1 and query.shape[-2] == key.shape[-2] and not module.training, 'Use full-prefix batch-one evaluation')
        index = module.layer_idx
        calls.append(index)
        if index in layers:
            attention_mask = receiver_mask(attention_mask, queries, sources)
        output, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if verify and index in layers and queries and sources:
            q, s = torch.tensor(queries, device=weights.device), torch.tensor(sources, device=weights.device)
            require(torch.count_nonzero(weights[:, :, q[:, None], s[None, :]]).item() == 0, 'Blocked weights are nonzero')
            rows = weights[:, :, q, :].float()
            require(torch.isfinite(rows).all().item() and torch.allclose(rows.sum(-1), torch.ones_like(rows.sum(-1)), atol=.01, rtol=0), 'Invalid attention normalization')
        return output, weights
    wrapped._medical_qa_wrapper = True
    impl.eager_attention_forward = wrapped
    try:
        yield
        require(sorted(calls) == list(range(shape.layers)), 'Expected one full forward per blocking context')
    finally:
        impl.eager_attention_forward = original

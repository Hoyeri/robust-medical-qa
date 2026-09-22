"""Model loading and prompt construction without server-specific dependencies."""
from dataclasses import dataclass
from pathlib import Path
from .io import require
from .spans import unique_span, overlapping


@dataclass(frozen=True)
class ModelShape:
    layers: int
    heads: int
    kv_heads: int
    head_dim: int

    @classmethod
    def from_model(cls, model):
        c = model.config
        shape = cls(c.num_hidden_layers, c.num_attention_heads, getattr(c, 'num_key_value_heads', c.num_attention_heads),
                    getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads)
        require(shape.heads % shape.kv_heads == 0, 'Incompatible grouped-query head counts')
        return shape


def load_runtime(model_path, tokenizer_path=None, device='cpu', dtype='float32'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    path = Path(tokenizer_path or model_path)
    if (path / 'tokenizer_config.json').exists():
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, use_fast=True)
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=str(path / 'tokenizer.json'))
        tok.bos_token = '<|begin_of_text|>'
        tok.eos_token = '<|eot_id|>'
        tok.chat_template = (path / 'chat_template.jinja').read_text()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, dtype=getattr(torch, dtype), attn_implementation='eager')
    model.to(device).eval()
    model.config.use_cache = False
    require(model.config.model_type == 'llama', 'Attention adapter currently supports Llama models')
    return model, tok


def add_runtime_arguments(parser):
    parser.add_argument('--model', required=True, help='Local model snapshot directory')
    parser.add_argument('--tokenizer', help='Local tokenizer directory; defaults to model directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')


def label_tokens(tok, prompt_config):
    prefix = tok(prompt_config['answer_prefix'], add_special_tokens=False)['input_ids']
    labels = {}
    for letter in prompt_config['choices']:
        whole = tok(prompt_config['answer_prefix'] + letter, add_special_tokens=False)['input_ids']
        require(whole[:-1] == prefix and len(whole) == len(prefix) + 1, f'Answer is not a separate token: {letter}')
        labels[letter] = whole[-1]
    require(len(set(labels.values())) == len(labels), 'Answer token IDs are not unique')
    return prefix, labels


def prompt_tokens(tok, question, options, config):
    require(set(options) == set(config['choices']), 'Option labels differ from prompt configuration')
    rendered = '\n'.join(f'{c}) {str(options[c]).strip()}' for c in config['choices'])
    user = f"{config['instruction']}\n\nQuestion:\n{str(question).strip()}\n\nOptions:\n{rendered}"
    messages = [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, 'get'):
        ids = ids['input_ids']
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    require(encoded['input_ids'] == ids, 'Chat retokenization mismatch')
    return ids, overlapping(encoded['offset_mapping'], unique_span(text, rendered)), text


def forward(model, ids, hidden=False):
    import torch
    return model(torch.tensor([ids], device=model.device), output_hidden_states=hidden, use_cache=False)

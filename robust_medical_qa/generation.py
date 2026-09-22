"""Greedy generation, answer parsing and exact-prefix replay."""
import json
import re
from .io import require
CHOICES = tuple("ABCD")

NATIVE_FINAL_MARKERS = ("## Final Response", "<unused95>")

_FENCED_JSON = re.compile(r"\A```json\s*(\{.*\})\s*```\Z", flags=re.IGNORECASE | re.DOTALL)

_ANSWER_FIELD = re.compile(r'["\']answer["\']\s*:\s*["\']([ABCD])["\']', flags=re.IGNORECASE)

_FINAL_ANSWER_FIELD = re.compile(r'["\']answer["\']\s*:\s*["\']([ABCD])["\']\s*\}?\s*\Z', flags=re.IGNORECASE)

def _final_section(rendered):
    located = []
    for marker in NATIVE_FINAL_MARKERS:
        idx = rendered.rfind(marker)
        if idx >= 0:
            located.append((idx, marker))
    if not located:
        return rendered, ""
    idx, marker = max(located)
    return rendered[idx + len(marker):].strip(), "native_final_response_"

def parse_compatible_response(text):
    """Full version (ported from Hamster's vendor.py, not the earlier
    pred_answer-only simplification) -- needed for answer_char_span/
    align_answer's rationale_structured_valid/parse_mode fields, added for
    post-hoc margin extraction on already-generated text (no regeneration)."""
    rendered = str(text).strip()
    candidate, prefix = _final_section(rendered)
    mode = f"{prefix}raw_json"
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
        mode = f"{prefix}single_markdown_json_fence"
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        fields = list(payload)
        rationale = payload.get("rationale")
        answer = str(payload.get("answer", "")).strip().upper()
        if fields == ["rationale", "answer"] and isinstance(rationale, str) and answer in CHOICES:
            return {
                "answer_valid": True,
                "rationale_structured_valid": True,
                "parse_mode": mode,
                "pred_answer": answer,
                "rationale": rationale,
                "field_order": fields,
            }
    final_answer = _FINAL_ANSWER_FIELD.search(candidate)
    json_shaped = candidate.startswith("{") and final_answer is not None
    matches = _ANSWER_FIELD.findall(candidate) if json_shaped else []
    if len(matches) == 1:
        return {
            "answer_valid": True,
            "rationale_structured_valid": False,
            "parse_mode": f"{prefix}malformed_json_unique_final_answer_field",
            "pred_answer": matches[0].upper(),
            "rationale": None,
            "field_order": None,
        }
    return {
        "answer_valid": False,
        "rationale_structured_valid": False,
        "parse_mode": "invalid",
        "pred_answer": None,
        "rationale": None,
        "field_order": None,
    }

def role(answer, row):
    return ('Invalid' if answer is None else 'G' if answer == row['gold_answer']
            else 'T' if answer == row['intended_target'] else 'Other')

def parse(text, row):
    parsed = parse_compatible_response(text)
    return {**parsed, 'scientific_role': role(parsed['pred_answer'], row)}

def answer_char_span(text, parsed):
    """Locate the top-level JSON field, never an answer mention inside rationale.

    Only strict JSON (optionally one markdown fence) with an unescaped one-letter
    answer is alignment-eligible. Other accepted parser modes remain behavioral
    observations, not guessed replay locations.
    """
    if not parsed['rationale_structured_valid'] or parsed['parse_mode'] not in (
            'raw_json', 'single_markdown_json_fence'):
        return None
    start = len(text)-len(text.lstrip()); end = len(text.rstrip())
    if parsed['parse_mode'] == 'single_markdown_json_fence':
        match = re.fullmatch(r'```json\s*(\{.*\})\s*```', text[start:end], re.I | re.S)
        if not match:
            return None
        start, end = start+match.start(1), start+match.end(1)
    decoder = json.JSONDecoder(); pos = start+1; fields = []
    spans = []
    try:
        while True:
            while text[pos].isspace(): pos += 1
            if text[pos] == '}': break
            key, pos = decoder.raw_decode(text, pos)
            while text[pos].isspace(): pos += 1
            if text[pos] != ':': return None
            pos += 1
            while text[pos].isspace(): pos += 1
            value_start = pos
            value, pos = decoder.raw_decode(text, pos)
            fields.append(key)
            if key == 'answer' and text[value_start:pos] == '"'+parsed['pred_answer']+'"':
                spans.append((value_start+1, value_start+2))
            while text[pos].isspace(): pos += 1
            if text[pos] == ',': pos += 1
            elif text[pos] == '}': break
            else: return None
    except (ValueError, IndexError, TypeError):
        return None
    if fields != ['rationale', 'answer'] or len(spans) != 1:
        return None
    return spans[0]

def align_answer(tokenizer, generated_ids, text, parsed, prompt_n):
    span = answer_char_span(text, parsed)
    if span is None:
        return {'status': 'unavailable', 'reason': 'no_unambiguous_strict_answer_field'}
    a, b = span
    previous = ''
    for offset in range(len(generated_ids)):
        current = tokenizer.decode(generated_ids[:offset+1], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        if len(previous) == a and len(current) == b and previous == text[:a] and current == text[:b]:
            if tokenizer.decode([generated_ids[offset]], clean_up_tokenization_spaces=False) == parsed['pred_answer']:
                return {'status': 'aligned', 'generated_token_offset': offset,
                        'answer_token_position': prompt_n+offset,
                        'prediction_position': prompt_n+offset-1,
                        'answer_token_id': generated_ids[offset], 'answer_char_span': [a, b]}
        previous = current
    return {'status': 'unavailable', 'reason': 'answer_not_a_separate_token_on_saved_path'}

def replay_offsets(n, alignment):
    if n < 1: raise ValueError('Empty generation')
    offsets = {0, n//2, n-1}
    if alignment['status'] == 'aligned': offsets.add(alignment['generated_token_offset'])
    return sorted(offsets)


def trace_logits(logits, offset, labels):
    import torch
    require(logits.ndim == 1 and torch.isfinite(logits).all().item(), 'Expected finite full-vocabulary logits')
    token = int(torch.argmax(logits))
    return {'generated_token_offset': offset, 'token_id': token, 'chosen_logit': float(logits[token]),
            'chosen_logprob': float(logits[token] - torch.logsumexp(logits, -1)),
            'candidate_logits': {label: float(logits[i]) for label, i in labels.items()}}


def generate(forward, tokenizer, prompt, row, config, labels):
    require(config['max_new_tokens'] > 0 and bool(config['stop_token_ids']), 'Invalid generation settings')
    ids, trace = [], []
    for offset in range(config['max_new_tokens']):
        step = trace_logits(forward(prompt + ids), offset, labels)
        ids.append(step['token_id'])
        trace.append(step)
        if ids[-1] in config['stop_token_ids']:
            break
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    parsed = parse(text, row)
    alignment = align_answer(tokenizer, ids, text, parsed, len(prompt))
    replays = []
    for offset in replay_offsets(len(ids), alignment):
        fresh = trace_logits(forward(prompt + ids[:offset]), offset, labels)
        require(fresh == trace[offset], 'Exact-prefix replay mismatch')
        replays.append({'offset': offset, 'trace': fresh, 'exact_equal': True})
    return dict(generated_token_ids=ids, generated_text=text, token_trace=trace,
                parsed=parsed, answer_alignment=alignment, replays=replays,
                finish_reason='stop' if ids[-1] in config['stop_token_ids'] else 'length')

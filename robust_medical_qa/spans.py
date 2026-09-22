"""Shared span extraction for existing variants and explicit span metadata."""
import re
from .io import require

TEMPLATE = re.compile(r"^(?:(?:She|He|The patient|Her coworker|His coworker|The patient's coworker|A classmate of (?:hers|his|the patient's)|Another (?:baby|child) at the clinic|Her parents|His parents|The patient's parents) "
                      r"(?:also has|has a history of|has had|has|previously had|reports a past history of|also complains of|additionally reports|reports|complains of|also have)\s+"
                      r"|(?:Examination also reveals|Examination additionally shows|Physical examination also demonstrates|Laboratory studies also show|Laboratory testing additionally reveals|Laboratory results also demonstrate|Additional laboratory studies show|Imaging also shows|Imaging additionally reveals|Imaging studies also demonstrate|There is a history of)\s+"
                      r"|(?:On further questioning, (?:she|he|the patient) reports)\s+|(?:On examination, |On imaging, )"
                      r"|(?:She|He|The patient) (?:denies|has no history of)\s+|(?:Examination reveals no|Laboratory studies show no|Imaging shows no)\s+)")


def normalize(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def unique_span(text, needle):
    require(bool(needle) and text.count(needle) == 1, 'Missing or ambiguous source string')
    start = text.index(needle)
    return start, start + len(needle)


def overlapping(offsets, span):
    a, b = span
    return [i for i, (x, y) in enumerate(offsets) if x < y and x < b and y > a]


def inserted_sentence(clean, variant):
    c, t = normalize(clean), normalize(variant)
    i = 0
    while i < min(len(c), len(t)) and c[i] == t[i]:
        i += 1
    j = 0
    while j < min(len(c), len(t)) - i and c[-1-j] == t[-1-j]:
        j += 1
    result = t[i:len(t)-j].strip()
    require(bool(result), 'No inserted sentence found')
    return result


def phrase_span_in(sentence):
    match = TEMPLATE.match(sentence)
    require(match is not None, 'Unknown finding template; supply explicit source_annotations')
    body = re.sub(r'\s+(is also noted|is also seen)$', '', sentence.rstrip('.').rstrip())
    require(match.end() < len(body), 'Empty finding phrase')
    return match.end(), len(body)


def finding_source(item, variant):
    annotation = item.get('source_annotations', {}).get(variant)
    if annotation is not None:
        sentence = annotation['sentence']
        a, b = annotation['phrase_char_span']
        require(0 <= a < b <= len(sentence), 'Invalid explicit finding span')
        return sentence, (a, b)
    variants = item['variants']
    # Historical duplicate condition reads the end-slot Ma finding.
    sentence = inserted_sentence(variants['C'], variants['Ma'] if variant == 'Mdup' else variants[variant])
    return sentence, phrase_span_in(sentence)


def source_positions(tokenizer, text, item, variant):
    sentence, (a, b) = finding_source(item, variant)
    require(sentence in text, f"{item['item_id']}/{variant}: source missing from prompt")
    if variant == 'Mdup':
        start = text.rindex(sentence)
    else:
        start, _ = unique_span(text, sentence)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    phrase = overlapping(offsets, (start + a, start + b))
    full = overlapping(offsets, (start, start + len(sentence)))
    require(bool(phrase) and bool(full) and 0 not in full, 'Invalid source token positions')
    return phrase, full, sentence[a:b], sentence


def attribution_metadata(ids, items):
    lookup = {it['item_id']: it for it in items}
    require(len(lookup) == len(items), 'Duplicate metadata item IDs')
    concepts, families = [], []
    for key in ids:
        item = lookup[key]
        concepts.append(item['meta']['cue']['hpo'])
        if 'attribution_family' in item.get('meta', {}):
            families.append(item['meta']['attribution_family'])
        else:
            sentence = inserted_sentence(item['variants']['C'], item['variants']['H1'])
            if 'coworker' in sentence:
                families.append('coworker')
            else:
                require('classmate' in sentence or 'Another child' in sentence or 'Another baby' in sentence,
                        'Unknown attribution family; set meta.attribution_family')
                families.append('classmate_child')
    return {'concept': concepts, 'family': families}


def nearest_length_sentence(tokenizer, clean, inserted):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', normalize(clean)) if s.strip()]
    if len(sentences) > 1:
        sentences = sentences[:-1]
    require(bool(sentences), 'No clinical control candidates')
    length = len(tokenizer(inserted, add_special_tokens=False)['input_ids'])
    return min(sentences, key=lambda s: (abs(len(tokenizer(s, add_special_tokens=False)['input_ids']) - length), sentences.index(s)))

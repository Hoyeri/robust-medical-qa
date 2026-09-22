"""Shared extraction loop for hidden-state and transmission experiments."""
import time
import os
from .checkpoints import atomic, read, sha
from pathlib import Path
import numpy as np
from .io import read_jsonl, write_json, digest, new_output_dir, require
from .runtime import label_tokens, prompt_tokens, ModelShape
from .spans import source_positions, nearest_length_sentence, overlapping, unique_span
from .representations import hidden_features, transmission_features


def extract(model, tok, items_path, output, config, prompt, kind, validate=0, limit=0, resume=False):
    require(kind in ('hidden', 'transmission'), 'Unknown extraction kind')
    rows = read_jsonl(items_path)
    if limit:
        rows = rows[:limit]
    require(len({r['item_id'] for r in rows}) == len(rows), 'Duplicate item IDs')
    out = Path(output) if resume else new_output_dir(output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / '.state'; cache.mkdir(exist_ok=True)
    prefix, labels = label_tokens(tok, prompt)
    shape = ModelShape.from_model(model)
    features = {v: {} for v in config['variants']}
    ids = {v: [] for v in features}
    phrases = {v: [] for v in features}
    alignment, validation, missing, controls = {}, [], [], []
    start = time.time()
    for n, item in enumerate(rows):
        alignment[item['item_id']] = {}
        for variant in config['variants']:
            if variant not in item['variants']:
                missing.append({'item_id': item['item_id'], 'variant': variant, 'reason': 'variant_absent'})
                continue
            prompt_ids, _, text = prompt_tokens(tok, item['variants'][variant], item['options'], prompt)
            phrase, full, phrase_text, sentence = source_positions(tok, text, item, variant)
            input_ids = prompt_ids + prefix
            key = sha([item['item_id'],variant])
            data_path, meta_path = cache/(key+'.npz'), cache/(key+'.json')
            binding = sha([item,variant,config,prompt,kind,validate,limit,str(model.dtype)])
            if meta_path.exists():
                meta = read(meta_path)
                require(meta['binding']==binding and sha(data_path)==meta['data'], 'Extraction checkpoint changed')
                with np.load(data_path, allow_pickle=False) as saved:
                    found = {k:saved[k] for k in saved.files}
                record,checks,control_errors = meta['record'],meta['checks'],meta['control_errors']
            else:
                record,checks,control_errors = None,[],[]
                if kind == 'hidden':
                    found = hidden_features(model, input_ids, phrase, config['pooling_dtype'])
                else:
                    sources = {'sent': full, 'phr': phrase}
                    if variant == config['control_variant']:
                        control = nearest_length_sentence(tok, item['variants']['C'], sentence)
                        try:
                            span = unique_span(text, control)
                        except ValueError as error:
                            control_errors.append({'item_id': item['item_id'], 'reason': str(error)})
                        else:
                            offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
                            sources['rand'] = overlapping(offsets, span)
                    validation_layers = config['validation_layers'] if n < validate and variant == config['control_variant'] else []
                    found, record, checks = transmission_features(model, input_ids, phrase, sources, labels,
                        item['gold_G'], item['target_T'], config['head_layers'], validation_layers, config['validation_alphas'])
                temp = data_path.with_suffix('.tmp')
                with temp.open('wb') as stream: np.savez_compressed(stream, **found)
                os.replace(temp,data_path)
                atomic(meta_path,dict(binding=binding,data=sha(data_path),record=record,checks=checks,control_errors=control_errors))
            if kind=='transmission':
                alignment[item['item_id']][variant]=record
                validation.extend({'item_id':item['item_id'],**check} for check in checks)
                controls.extend(control_errors)
            for stage, array in found.items():
                features[variant].setdefault(stage, []).append(array.astype(config['storage_dtype']))
            ids[variant].append(item['item_id'])
            phrases[variant].append(phrase_text)
        print(f'{n+1}/{len(rows)} extracted', flush=True)
    for variant, stages in features.items():
        if ids[variant]:
            np.savez_compressed(out / f'{variant}.npz', **{k: np.stack(v) for k, v in stages.items()},
                                item_ids=np.array(ids[variant]), phrases=np.array(phrases[variant]))
    if kind == 'transmission':
        atomic(out / 'alignment.json', alignment)
        atomic(out / 'validation.json', validation)
    atomic(out / 'RUN_INFO.json', dict(kind=kind, n=len(rows), variants=config['variants'],
        completed_by_variant={k: len(v) for k, v in ids.items()}, missing_variants=missing,
        missing_controls=controls, config=config, prompt_config=prompt, model_shape=shape.__dict__,
        model_path=str(model.config._name_or_path), model_dtype=str(model.dtype), device=str(model.device),
        label_ids=labels, items_sha256=digest(items_path), seconds=time.time()-start))

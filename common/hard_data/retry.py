"""Bounded option-cue replenishment, independent of inference and score ranking.

Each initial candidate owns a slot. Only a cue-only rejection reopens that slot.
Technical failures are terminal. Content is NOT an additional acceptance gate.
Saved generation and each judgment are
replayed on resume; completed inference is never rerun when its checkpoint exists.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import fcntl

from common.checkpoints import Checkpoints, sha


@dataclass(frozen=True)
class RetryPolicy:
    slots: int = 8
    extra_rounds: int = 2
    base_seed: int = 42
    namespace: str = 'nl-cue-retry-v2'

    def __post_init__(self):
        if not 1 <= self.slots <= 8 or not 0 <= self.extra_rounds <= 2:
            raise ValueError('Supported budget: 1..8 slots, 0..2 extra rounds')
        if not 0 <= self.base_seed < 2**31 or not self.namespace:
            raise ValueError('Invalid seed or namespace')


def round_seed(policy, source_id, target, round_index):
    if not 0 <= round_index <= policy.extra_rounds:
        raise ValueError('Round outside approved budget')
    seeds = [policy.base_seed]
    for index in range(1, round_index + 1):
        seed = int(sha([policy.namespace, policy.base_seed, source_id, target, index])[:16], 16) % (2**31)
        while seed in seeds:
            seed = (seed + 1) % (2**31)
        seeds.append(seed)
    return seeds[-1]


def candidate_id(policy, source_id, target, slot, round_index):
    return f'{policy.namespace}:{source_id}:{target}:s{slot}:r{round_index}'


def classify_failure(candidate, judgments, target, gold, gate):
    """Use only the bystander option-cue decision after technical validation."""
    if (not candidate.get('sentence', '').strip()
            or '<|channel|>' in candidate['sentence']
            or candidate.get('finish_reason') != 'stop'):
        return dict(category='technical', retryable=False, reasons=['GENERATION_INCOMPLETE_OR_EMPTY'], gate=None)
    technical = []
    for order in ('forward', 'reverse'):
        item = judgments.get(order, {})
        if item.get('parsed') is None or item.get('error') or item.get('finish_reason') != 'stop':
            technical.append(order + ':JUDGMENT_INCOMPLETE_OR_INVALID')
    if technical:
        return dict(category='technical', retryable=False, reasons=technical, gate=None)
    result = gate(judgments, target, gold)
    if result['pass']:
        return dict(category='accepted', retryable=False, reasons=[], gate=result)
    allowed = {'NO_STRENGTH', 'GOLD_IN_MULTI', 'GOLD_TARGET', 'NONE_TARGET', 'OTHER',
               'INTENDED_SINGLE_TARGET', 'MULTI_TARGET_WRONG_ONLY'}
    structures = [result['forward_structure'], result['reverse_structure']]
    if not set(structures) <= allowed:
        raise ValueError(f'Unclassified gate failure: {structures}')
    return dict(category='option_cue', retryable=True, reasons=structures, gate=result)


def run_target(source, target, generation_template, judge_template, policy,
               checkpoint_dir, generate, judge, gate, runtime_binding,
               check=lambda: None, progress=lambda event: None):
    """Callbacks: generate(prompt,n,seed)->list; judge(prompt)->parsed+raw dict.

    No correctness, confidence, difficulty, or score enters the retry decision.
    Each callback must preserve raw output and finish_reason. Schema validation
    of judge output belongs to the unchanged legacy parser used by the backend.
    """
    if target not in 'ABCD' or target == source['answer_idx']:
        raise ValueError('Target must be an original wrong choice')
    if set(source['options']) != set('ABCD'):
        raise ValueError('Expected original A-D options')
    store = Checkpoints(checkpoint_dir)
    sid = source['source_id']
    job = sid + ':' + target
    lock = (store.root / (sha(job) + '.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(source=source, target=target, generation_template=generation_template,
                       judge_template=judge_template, policy=asdict(policy), runtime=runtime_binding,
                       gate_contract='bystander_option_cue_v1_no_content_gate')
        store.cached('binding:' + job, binding, lambda: binding)
        prompt = generation_template.format(question=source['question'], clinical_topic=source['options'][target])
        active = list(range(policy.slots))
        attempts, accepted, slots = [], [], {}
        for round_index in range(policy.extra_rounds + 1):
            if not active:
                break
            check()
            seed = round_seed(policy, sid, target, round_index)
            inputs = dict(binding=sha(binding), prompt=prompt, n=len(active), slots=active, seed=seed)
            batch = store.cached(f'generate:{job}:r{round_index}', inputs,
                                 lambda: generate(prompt, len(active), seed))
            if len(batch) != len(active):
                raise ValueError('Generation returned the wrong batch size')
            again = []
            for slot, output in zip(active, batch):
                check()
                cid = candidate_id(policy, sid, target, slot, round_index)
                can = dict(output, candidate_id=cid, source_id=sid, source_idx=source['idx'],
                           intended_target=target, slot=slot, round=round_index,
                           candidate_rank=slot * (policy.extra_rounds + 1) + round_index,
                           seed=seed, prompt_sha256=sha(prompt))
                can['candidate_text_sha256'] = __import__('hashlib').sha256(can['sentence'].encode()).hexdigest()
                judgments = {}
                if can['sentence'].strip() and can['finish_reason'] == 'stop' and '<|channel|>' not in can['sentence']:
                    for order, tag in [('ABCD', 'forward'), ('DCBA', 'reverse')]:
                        check()
                        jprompt = judge_template.format(question=source['question'], sentence=can['sentence'],
                            choices='\n'.join(f'{k}. {source["options"][k]}' for k in order))
                        judgments[tag] = store.cached(f'judge:{cid}:{order}',
                            dict(binding=sha(binding), prompt=jprompt, seed=policy.base_seed),
                            lambda p=jprompt: judge(p))
                decision = classify_failure(can, judgments, target, source['answer_idx'], gate)
                record = dict(candidate=can, judgments=judgments, decision=decision)
                saved = store.cached('decision:' + cid, record, lambda r=record: r)
                attempts.append(saved)
                status = decision['category']
                if status == 'accepted':
                    accepted.append(dict(can, generation_valid=True, judgments=judgments,
                                         target_gate_label=decision['gate']['label']))
                elif decision['retryable'] and round_index < policy.extra_rounds:
                    again.append(slot)
                elif decision['retryable']:
                    status = 'option_cue_budget_exhausted'
                slots[slot] = dict(slot=slot, final_candidate_id=cid, status=status,
                                   attempts=round_index + 1)
            progress(dict(source_id=sid, target=target, round=round_index, generated=len(batch),
                          accepted_total=len(accepted), next_round_n=len(again)))
            active = again
        result = dict(source_id=sid, target=target, policy=asdict(policy),
                      attempts=attempts, valid_candidates=accepted,
                      slots=[slots[s] for s in range(policy.slots)], generated_n=len(attempts),
                      accepted_n=len(accepted), shortage_n=policy.slots-len(accepted))
        if len(attempts) > policy.slots * (policy.extra_rounds + 1) or len(accepted) > policy.slots:
            raise ValueError('Retry budget violated')
        store.cached('complete:' + job, result, lambda: result)
        return result
    finally:
        lock.close()


def build_scoring_pool(sources, results, insert, sample, condition="nonliteral", random_salt="nl-random-valid-v1"):
    """The SAME frozen accepted pool feeds score requests and Random/Hard."""
    index = {r['source_id']: r for r in sources}
    if len(index) != len(sources):
        raise ValueError('Duplicate source IDs')
    jobs = [(r['source_id'], r['target']) for r in results]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate source-target results')
    valid, requests, random = [], [], {}
    def request(row, question, can=None):
        return dict(request_id=can['candidate_id'] if can else 'clean:'+row['source_id'],
                    source_id=row['source_id'], source_idx=row['idx'], role='candidate' if can else 'clean',
                    condition=condition if can else 'clean', question=question, options=row['options'],
                    gold=row['answer_idx'], candidate_id=can['candidate_id'] if can else None,
                    intended_target=can['intended_target'] if can else None)
    for source in sources:
        requests.append(request(source, source['question']))
    for result in results:
        row = index[result['source_id']]
        for can in result['valid_candidates']:
            if can['source_id'] != row['source_id'] or can['intended_target'] != result['target']:
                raise ValueError('Candidate provenance mismatch')
            valid.append(can)
            requests.append(request(row, insert(row['question'], can['sentence']), can))
    if len({r['candidate_id'] for r in valid}) != len(valid):
        raise ValueError('Duplicate accepted candidate IDs')
    for sid in index:
        pool = [r for r in valid if r['source_id'] == sid]
        if pool:
            random[sid] = sample(pool, sid, random_salt)['candidate_id']
    return dict(valid_candidates=valid, score_requests=requests, random_selection=random,
                source_coverage_n=len(random), source_n=len(sources))

"""GPU backend for the reusable cue retry engine; no policy decisions here."""
import importlib.metadata
from common.checkpoints import sha
from .scoring import DEFAULT_MODEL as MODEL, DEFAULT_REVISION as REVISION
from .judge import extract_final
from .judge import parse_judgment


class VllmRetryBackend:
    def __init__(self, chat_date="2026-09-12"):
        self.chat_date = chat_date
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        if importlib.metadata.version('vllm') != '0.24.0':
            raise RuntimeError('Unexpected vLLM version')
        self.params = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                       trust_remote_code=True, local_files_only=True)
        self.llm = LLM(model=MODEL, revision=REVISION,
                       tokenizer_revision=REVISION, trust_remote_code=True,
                       tensor_parallel_size=1, max_model_len=8192,
                       gpu_memory_utilization=.85, seed=42, max_num_seqs=8)
        self.extract = extract_final
        self.calls = dict(generate=0, judge=0)

    def render(self, prompt):
        rendered = self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],
            tokenize=False,add_generation_prompt=True,reasoning_effort='low')
        # Freeze the automatic Harmony date across restarts.
        import re
        return re.sub(r'Current date: \d{4}-\d{2}-\d{2}', 'Current date: ' + self.chat_date, rendered)

    def generate(self, prompt, n, seed):
        self.calls['generate'] += 1
        rendered = self.render(prompt)
        output = self.llm.generate([rendered], self.params(n=n,temperature=.7,top_p=.95,
            seed=seed,max_tokens=512,skip_special_tokens=False),use_tqdm=False)[0].outputs
        return [dict(sentence=self.extract(r.text),raw_generation=r.text,
                     generated_token_ids=list(r.token_ids),finish_reason=r.finish_reason,
                     rendered_prompt_sha256=sha(rendered)) for r in output]

    def judge(self, prompt):
        self.calls['judge'] += 1
        rendered = self.render(prompt)
        r = self.llm.generate([rendered],self.params(n=1,temperature=0,seed=42,
            max_tokens=512,skip_special_tokens=False),use_tqdm=False)[0].outputs[0]
        parsed,error = parse_judgment(r.text)
        return dict(parsed=parsed,error=error,raw=r.text,tokens=list(r.token_ids),
                    finish_reason=r.finish_reason,rendered_prompt_sha256=sha(rendered))

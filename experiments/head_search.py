"""Run the original head-search scripts with their complete matching runtime."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import os
import shutil
import subprocess
from common.checkpoints import Run, sha

SNAPSHOT=ROOT/'common/vendor/lsld-v1.0.0-medqa'
REQUIRED=('data.py','io_utils.py','paper_head_data.py','protocol.py','tokenization.py',
          'official_code/head_search/head_analysis.py',
          'official_code/head_search/circuit_lms/transformer_blocks.py')


def main():
    parser=argparse.ArgumentParser(description='Run the original head-mask search, mask evaluation, and individual-head evaluation')
    parser.add_argument('--input',required=True,help='Directory with train/dev/test JSONL, 89/11/11 items')
    parser.add_argument('--code-dir','--runtime',dest='runtime',metavar='PATH',required=True,
                        help='Complete lsld-v1.0.0 code directory supplied by janthonio03, containing src/lsld_repro')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    runtime=Path(args.runtime).resolve();data=Path(args.input).resolve()
    missing=[name for name in REQUIRED if not (runtime/'src/lsld_repro'/name).is_file()]
    if missing:
        parser.error('Complete lsld-v1.0.0 runtime required. Missing: '+', '.join(missing))
    # Refuse mixing a different paper_head implementation with the matching runtime.
    for relative in ('src/lsld_repro/paper_head.py',
        'src/lsld_repro/official_code/head_search/circuit_lms/hooked_transformers.py'):
        other=runtime/relative
        if not other.exists() or sha(other)!=sha(SNAPSHOT/relative):
            parser.error('The supplied runtime does not match the imported medical snapshot: '+relative)
    binding=dict(data={s:sha(data/(s+'.jsonl')) for s in ('train','dev','test')},
                 runtime={str(p.relative_to(runtime)):sha(p) for p in sorted((runtime/'src').rglob('*.py'))})
    with Run(args.output,binding,args.resume) as run:
        work=run.state/'runtime'
        if not work.exists():
            shutil.copytree(runtime/'src',work/'src')
            shutil.copytree(SNAPSHOT/'scripts',work/'scripts')
        env=dict(os.environ,PYTHONPATH=str(work/'src')+os.pathsep+str(work/'scripts'))
        def execute(script,destination,extra=()):
            subprocess.run([sys.executable,str(work/'scripts'/script),'--data-dir',str(data),
                '--output-dir',str(destination),*map(str,extra)],env=env,check=True)
            return [p for p in destination.rglob('*') if p.is_file()]
        training=run.state/'training'
        def train():
            if training.exists() and any(training.iterdir()):
                raise RuntimeError('Original trainer has no optimizer-resume support. Preserve this run and use a new output directory for an interrupted search.')
            return execute('run_medqa_head_search.py',training)
        run.stage('search',train)
        run.stage('mask_evaluation',lambda:execute('evaluate_medqa_all111_mask.py',run.output/'mask',
                                                   ['--best-mask',training/'best_mask.json']))
        run.stage('head_scan',lambda:execute('scan_medqa_all111_heads.py',run.output/'heads'))
        for name in ('head_search_report.json','best_mask.json'):
            shutil.copy2(training/name,run.output/name)
        print('Head analysis complete: '+str(run.output))


if __name__ == "__main__":
    main()

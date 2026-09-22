"""Atomic experiment state and checkpoint validation."""
import os
import json
import hashlib
import fcntl
from pathlib import Path


def sha(value):
    if isinstance(value, Path): value = value.read_bytes()
    elif not isinstance(value, bytes): value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    os.replace(temp, path)


def sample_by_hash(pool, key, salt):
    if not pool: raise ValueError('Empty sampling pool')
    return min(pool, key=lambda c: sha([salt, key, c['candidate_id']]))

class Checkpoints:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def cached(self, key, inputs, action):
        path = self.root / (sha(key) + '.json')
        input_hash = sha(inputs)
        if path.exists():
            value = read(path)
            if value['key'] != key or value['input_hash'] != input_hash:
                raise ValueError('Checkpoint input drift: ' + key)
            if value['data_hash'] != sha(value['data']):
                raise ValueError('Checkpoint data corruption: ' + key)
            return value['data']
        data = action()
        atomic(path, dict(key=key, input_hash=input_hash, data=data, data_hash=sha(data)))
        return data



class Run:
    """Hold a single-writer lock, bind inputs/config/code, verify completed stages."""
    def __init__(self, output, binding, resume=False):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()) and not resume:
            raise ValueError('Output is not empty; use --resume or a new output directory')
        self.state = self.output / '.state'
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = (self.state / 'run.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            root = Path(__file__).resolve().parent
            code = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                    if p.suffix in ('.py', '.json', '.yaml') and 'tests' not in p.parts and '__pycache__' not in p.parts}
            code.update({'experiments/' + p.name: sha(p) for p in sorted((root.parent / 'experiments').glob('*.py'))})
            Checkpoints(self.state).cached('run', dict(binding=binding, code=code), lambda: True)
        except Exception:
            self.lock.close()
            raise

    def stage(self, name, action):
        marker = self.state / (name + '.json')
        if marker.exists():
            value = read(marker)
            for rel, digest in value['files'].items():
                path = self.output / rel
                if not path.is_file() or sha(path) != digest:
                    raise ValueError('Completed output changed: ' + rel)
            return
        paths = action()
        atomic(marker, {'files': {str(Path(p).relative_to(self.output)): sha(Path(p)) for p in paths}})

    def __enter__(self): return self
    def __exit__(self, *args): self.lock.close()


def child_environment(**overrides):
    root=str(Path(__file__).resolve().parents[1])
    env=dict(os.environ,**overrides)
    env['PYTHONPATH']=root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env

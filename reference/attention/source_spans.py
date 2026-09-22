"""Exact source attribution positions; no scores or model-based selection."""
import hashlib
from run_e01 import require


def unique_span(text,needle):
    require(bool(needle) and text.count(needle)==1,'Missing/ambiguous exact source string')
    a=text.index(needle)
    return a,a+len(needle)


def overlapping(offsets,span):
    a,b=span
    if a==b: return []
    return [i for i,(x,y) in enumerate(offsets) if x<y and x<b and y>a]


def matched_subset(indices,n,question_id):
    n=min(n,len(indices))
    if not n: return []
    start=int(hashlib.sha256(question_id.encode()).hexdigest()[:16],16)%(len(indices)-n+1)
    return indices[start:start+n]

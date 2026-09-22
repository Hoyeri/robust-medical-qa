"""Pure selection and growing-prefix receiver policy for natural generation."""
import hashlib
from run_e01 import require

POLICIES=['baseline','option_cut','all_source_cut']

def select_ids(ids,n=96):
    require(len(ids)==len(set(ids)) and len(ids)>=n,'Unique eligible IDs')
    return sorted(ids,key=lambda q:hashlib.sha256(('e08-natural-v1:'+q).encode()).hexdigest())[:n]

def generation_queries(policy,n,source,options):
    require(policy in POLICIES and source and all(0<p<n for p in source),'Generation policy/source')
    require(not set(source)&set(options) and all(0<=p<n for p in options),'Option receivers')
    if policy=='baseline':return []
    if policy=='option_cut':return list(options)
    return [p for p in range(min(source)+1,n) if p not in set(source)]

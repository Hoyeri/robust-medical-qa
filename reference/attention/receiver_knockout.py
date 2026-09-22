"""Pre-softmax knockout for explicit receiver/source and layer sets."""
from contextlib import contextmanager
from run_e01 import require


def positions_ok(values,n):
    require(len(values)==len(set(values)) and all(type(x) is int and 0<=x<n for x in values),'Unique in-bounds positions')


def receiver_mask(mask,queries,sources,torch):
    require(mask is not None and mask.ndim==4 and mask.shape[0]==1 and mask.shape[-1]==mask.shape[-2],'Full batch1 explicit mask')
    n=mask.shape[-1];positions_ok(queries,n);positions_ok(sources,n)
    require(not set(queries)&set(sources),'Source and receivers must be disjoint')
    require(0 not in sources and len(sources)<n,'Keep causal key0 available')
    if not queries or not sources:return mask
    q=torch.tensor(queries,device=mask.device);s=torch.tensor(sources,device=mask.device)
    result=mask.clone();result[:,:,q[:,None],s[None,:]]=float('-inf')
    return result


@contextmanager
def receiver_edges(torch,layers,queries,sources):
    from transformers.models.llama import modeling_llama as impl
    original=impl.eager_attention_forward
    require(not getattr(original,'_discovery_wrapper',False),'Nested attention wrapper')
    require(len(layers)==len(set(layers)) and all(type(x) is int and 0<=x<32 for x in layers),'Layer set')
    info={'calls':[],'selected_calls':[],'zero_verified_layers':[]}
    def wrapped(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        index=module.layer_idx;info['calls'].append(index)
        require(query.shape[0]==1 and query.shape[1]==32 and key.shape[1]==8 and value.shape[1]==8,'GQA shape')
        require(query.shape[-2]==key.shape[-2] and dropout==0 and not module.training,'Full inference only')
        if index in layers:
            info['selected_calls'].append(index)
            attention_mask=receiver_mask(attention_mask,queries,sources,torch)
        output,weights=original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        if index in layers and queries and sources:
            q=torch.tensor(queries,device=weights.device);s=torch.tensor(sources,device=weights.device)
            require(torch.count_nonzero(weights[:,:,q[:,None],s[None,:]]).item()==0,'Selected edge weights zero')
            rows=weights[:,:,q,:].float()
            require(torch.isfinite(rows).all().item() and torch.allclose(rows.sum(-1),torch.ones_like(rows.sum(-1)),atol=.01,rtol=0),'Selected rows finite normalized')
            info['zero_verified_layers'].append(index)
        return output,weights
    wrapped._discovery_wrapper=True;impl.eager_attention_forward=wrapped
    try:
        yield info
        require(sorted(info['calls'])==list(range(32)) and sorted(info['selected_calls'])==sorted(layers),'Layer call inventory')
        require(not queries or not sources or sorted(info['zero_verified_layers'])==sorted(layers),'Zero checks inventory')
    finally:impl.eager_attention_forward=original


def toy_tests(t):
    m=t.triu(t.full((1,1,5,5),float('-inf')),diagonal=1)
    old=m.clone();z=receiver_mask(m,[2,4],[1,3],t)
    expected=m.clone()
    for q in [2,4]:
        for s in [1,3]:expected[0,0,q,s]=float('-inf')
    require(t.equal(z,expected) and t.equal(m,old),'Cross-product and immutable original')
    w=t.softmax(z,dim=-1)
    require(t.isfinite(w).all() and t.allclose(w.sum(-1),t.ones(1,1,5)),'Finite normalized rows')
    require(w[0,0,2,1]==0 and w[0,0,4,3]==0 and w[0,0,2,4]==0,'Source and causal zeros')
    require(receiver_mask(m,[4],[],t) is m and receiver_mask(m,[],[1],t) is m,'Empty identity')
    for q,s in [([4,4],[1]),([4],[1,1]),([5],[1]),([4],[0]),([2],[2])]:
        try:receiver_mask(m,q,s,t)
        except ValueError:pass
        else:raise AssertionError('Bad positions accepted')

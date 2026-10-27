"""Pinned Qwen27B GQA-6 attention probe; isolated from serving model."""
import importlib.util,json,statistics,sys,time
from pathlib import Path
import torch,triton
from vllm.v1.attention.ops import triton_unified_attention as native

path=Path(__file__).with_name('attention-candidate-unmasked.py')
spec=importlib.util.spec_from_file_location('amd_attn_candidate',path)
candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)

class Capture:
    def __init__(self,fn):self.fn=fn;self.last=None
    def __getitem__(self,grid):
        launch=self.fn[grid]
        def call(*args,**kwargs):self.last=launch(*args,**kwargs);return self.last
        return call
capture=Capture(candidate.kernel_unified_attention)
candidate.kernel_unified_attention=capture

def timed(fn):
    times=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
    return times

torch.set_num_threads(1);torch.manual_seed(20261027)
rows=[]
with torch.inference_mode():
  for M,ctx in ((1536,6656),):
    H,HK,D,B=24,4,256,832;S=ctx+M;NB=triton.cdiv(S,B)
    q=torch.randn(M,H,D,device='cuda',dtype=torch.float16)
    # Match a head-before-token physical cache, exposed as [block,token,head,dim].
    k=torch.randn(NB,HK,B,D,device='cuda',dtype=torch.float16).permute(0,2,1,3)
    v=torch.randn_like(k)
    table=torch.randperm(NB,device='cuda').to(torch.int32)[None,:]
    flat_k=k[table[0].long()].reshape(-1,HK,D)[:S].repeat_interleave(H//HK,dim=1).float()
    flat_v=v[table[0].long()].reshape(-1,HK,D)[:S].repeat_interleave(H//HK,dim=1).float()
    qi=torch.unique(torch.cat((torch.linspace(0,M-1,24,device='cuda').long(),torch.tensor([1,2,3,30,31,32,33],device='cuda'))))
    scores=torch.einsum('qhd,khd->hqk',q[qi].float(),flat_k)*(D**-.5)
    scores.masked_fill_(torch.arange(S,device='cuda')[None,None,:]>ctx+qi[None,:,None],float('-inf'))
    reference=torch.einsum('hqk,khd->qhd',scores.softmax(dim=-1),flat_v).half()
    del flat_k,flat_v,scores
    out=torch.empty_like(q);baseline=torch.empty_like(q)
    kwargs=dict(q=q,k=k,v=v,cu_seqlens_q=torch.tensor([0,M],device='cuda',dtype=torch.int32),
        max_seqlen_q=M,seqused_k=torch.tensor([S],device='cuda',dtype=torch.int32),max_seqlen_k=S,
        softmax_scale=D**-.5,causal=True,window_size=(-1,-1),block_table=table,
        softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
    def original():native.unified_attention(out=baseline,**kwargs)
    original();torch.cuda.synchronize()
    native_err=(baseline[qi].float()-reference.float()).norm()/reference.float().norm()
    assert torch.isfinite(baseline).all() and native_err<.003,float(native_err)
    for cfg in ((16,32,None,None),(16,32,4,1),(128,64,8,1)):
        candidate._AMD_TEST_CONFIG=cfg
        def run():candidate.unified_attention(out=out,**kwargs)
        row=dict(M=M,context_tokens=ctx,query_heads=H,kv_heads=HK,head_dim=D,physical_block=B,
                 tile=cfg,native_reference_relative_l2=float(native_err))
        try:
            start=time.perf_counter();run();torch.cuda.synchronize();row['first_compile_s']=time.perf_counter()-start
            err=(out[qi].float()-reference.float()).norm()/reference.float().norm()
            delta=(out.float()-baseline.float()).norm()/baseline.float().norm()
            assert torch.isfinite(out).all() and err<.003 and delta<.003,(float(err),float(delta))
            row.update(reference_relative_l2=float(err),native_relative_l2=float(delta),
                       n_regs=capture.last.n_regs,n_spills=capture.last.n_spills,shared_bytes=capture.last.metadata.shared)
            before=timed(original);times=timed(run);after=timed(original)
            row.update(before_native_ms=before,candidate_ms=times,after_native_ms=after,
                       speedup=statistics.median(before+after)/statistics.median(times),numerics_passed=True)
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:600]
        rows.append(row);print(json.dumps(row),flush=True)
    del q,k,v,table,qi,reference,out,baseline,kwargs
    torch.cuda.empty_cache()
print('RESULT_JSON='+json.dumps({'scope':'Random real-shape per-layer causal GQA6 attention. HIP event timings, independent FP32 softmax reference; not serving performance. Candidate additionally masks padded query rows to give each output one writer.','rows':rows}),flush=True)

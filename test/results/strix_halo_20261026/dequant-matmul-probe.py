"""Bounded prefill study: transient FP16 dequant + rocBLAS versus fused tiles."""
import json, statistics
import torch, triton
import triton.language as tl
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as rdna

@triton.jit
def dequant(B,S,Z,W,K:tl.constexpr,N:tl.constexpr,G:tl.constexpr,BS:tl.constexpr,T:tl.constexpr):
    flat=tl.program_id(0)*T+tl.arange(0,T)
    n,k=flat//K,flat%K
    packed=tl.load(B+n*BS+k//8,n<N,0)
    shift=(k%8//2)*4+(k%2)*16
    q=(packed>>shift)&15
    z=(tl.load(Z+(n//8)*(K//G)+k//G,n<N,0)>>(4*(n%8)))&15
    scale=tl.load(S+n*(K//G)+k//G,n<N,1)
    w=(q-z).to(scale.dtype)*scale
    tl.store(W+flat,w,n<N)

def timed(fn):
    samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return samples

torch.set_num_threads(1);torch.manual_seed(20261027)
rdna._AMD_QWEN27B_PREFILL=True
rows=[]
with torch.inference_mode():
  for M,K,N in [(2496,5120,34816),(2496,5120,16384),(2496,6144,5120),(2496,17408,5120),(1600,5120,34816)]:
    G=32;pad=32 if (K//2)%1024==0 else 0
    a=rdna._pad_activation_rows(torch.randn(M,K,device='cuda',dtype=torch.float16))
    b=torch.randint(-(2**31),2**31-1,(N,K//8+pad),device='cuda',dtype=torch.int32)[:,:K//8]
    s=torch.rand(N,K//G,device='cuda',dtype=torch.float16)*.01+.001
    z=torch.randint(-(2**31),2**31-1,(N//8,K//G),device='cuda',dtype=torch.int32)
    w=torch.empty(N,K,device='cuda',dtype=torch.float16)
    y=torch.empty(M,N,device='cuda',dtype=torch.float16)
    def dq():dequant[(triton.cdiv(N*K,1024),)](b,s,z,w,K,N,G,b.stride(0),1024)
    def fused():return rdna.triton_w4a16_skinny_fmt_gemm(a,b,s,G,zp=z)
    def separate():dq();torch.mm(a,w.t(),out=y)
    separate();reference=fused();torch.cuda.synchronize()
    error=(reference.float()-y.float()).norm()/reference.float().norm()
    assert torch.isfinite(y).all() and error<.002,float(error)
    control1=timed(fused);dqtime=timed(dq);mmtime=timed(lambda:torch.mm(a,w.t(),out=y));both=timed(separate);control2=timed(fused)
    row=dict(M=M,K=K,N=N,relative_l2=float(error),fused_before_ms=control1,
             dequant_ms=dqtime,matmul_ms=mmtime,separate_ms=both,fused_after_ms=control2,
             speedup=statistics.median(control1+control2)/statistics.median(both),
             transient_weight_bytes=w.numel()*w.element_size())
    rows.append(row);print(json.dumps(row),flush=True)
    del a,b,s,z,w,y,reference
    torch.cuda.empty_cache()
print('RESULT_JSON='+json.dumps({'scope':'Per-layer random tensor probe, includes dequant on every call; not serving performance','rows':rows}),flush=True)

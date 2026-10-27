"""Bounded group32 FFN tile probe; never modifies the running model.

Inspired by Gufo's large blocked WMMA prefill. Uses the installed vLLM
kernel and random asymmetric AWQ-format tensors. Not a serving benchmark.
"""
import json
import statistics
import time
import torch
import triton
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as rdna

torch.set_num_threads(1)
torch.manual_seed(20261027)
rows = []
M, G = 4096, 32
report = {"scope": "Four actual Qwen27B projection shapes at observed chunk lengths, random tensors, warm GPU-event timing; no end-to-end claim", "rows": rows}

def timed(call):
    samples=[]
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(); call(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples

with torch.inference_mode():
    for M, K, N in [(m,k,n) for m in (1600,2496) for k,n in ((5120,34816),(5120,16384),(5120,14336),(6144,5120))]:
        a = rdna._pad_activation_rows(torch.randn(M, K, device="cuda", dtype=torch.float16))
        pad = 32 if (K//2)%1024 == 0 else 0
        b = torch.randint(-(2**31), 2**31-1, (N, K//8+pad), device="cuda", dtype=torch.int32)[:, :K//8]
        s = torch.rand(N, K//G, device="cuda", dtype=torch.float16)*0.01+0.001
        z = torch.randint(-(2**31), 2**31-1, (N//8, K//G), device="cuda", dtype=torch.int32)
        out = torch.empty(M, N, device="cuda", dtype=torch.float16)
        # Independent FP32 reference for spread-out output columns.
        nk = torch.linspace(0,N-1,32,device="cuda").long()
        kk = torch.arange(K,device="cuda")
        shift = (kk%8//2)*4+(kk%2)*16
        q = (b[nk[:,None],kk[None,:]//8] >> shift[None,:]) & 15
        zero = (z[nk[:,None]//8,kk[None,:]//G] >> (4*(nk[:,None]%8))) & 15
        w = (q-zero).half()*s[nk[:,None],kk[None,:]//G]
        ref = (a.float() @ w.float().t()).half()
        del q, zero, w
        native = (128,64,32,8) if K<2*N else (128,512,32,16)
        configs = [native, (256,256,32,32)]
        # Each arm bracketed by the same native control to expose drift.
        def launch(cfg):
            bm,bn,bk,nw=cfg
            return rdna._triton_w4a16_skinny_fmt_kernel[(triton.cdiv(M,bm),triton.cdiv(N,bn))](
                a,b,s,z,out,M,N,K,K//8,K//G,b.stride(0),a.stride(0),group_size=G,
                ZP_BIAS=8,HAS_ZP=True,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,num_warps=nw)
        for cfg in configs:
            record={"M":M,"K":K,"N":N,"group_size":G,"tile":cfg,"native":cfg==native}
            try:
                t=time.perf_counter(); compiled=launch(cfg);torch.cuda.synchronize()
                record["compile_and_first_s"]=time.perf_counter()-t
                sampled=out[:,nk]
                relative=(sampled.float()-ref.float()).norm()/ref.float().norm()
                assert torch.isfinite(out).all() and relative<0.0015, float(relative)
                record.update(numerics_passed=True,reference_relative_l2=float(relative),
                              n_regs=compiled.n_regs,n_spills=compiled.n_spills,
                              shared_bytes=compiled.metadata.shared)
                launch(native); torch.cuda.synchronize()
                record["before_native_ms"]=timed(lambda:launch(native))
                record["candidate_ms"]=timed(lambda:launch(cfg))
                record["after_native_ms"]=timed(lambda:launch(native))
                control=statistics.median(record["before_native_ms"]+record["after_native_ms"])
                record["speedup"]=control/statistics.median(record["candidate_ms"])
            except Exception as exc:
                record["error"]=type(exc).__name__+": "+str(exc)[:500]
            rows.append(record)
            print(json.dumps(record),flush=True)
        del a,b,s,z,out,ref,nk,kk,shift,sampled
        torch.cuda.empty_cache()
print("RESULT_JSON="+json.dumps(report),flush=True)

"""GPU gate for the optional Qwen27B tile: boundaries, tails and FP32 oracle.

Run inside the patched v0.32 ROCm image. It toggles only this process's
dispatch flag; no running inference process is modified.
"""
import json
import torch
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as rdna

torch.set_num_threads(1)
torch.manual_seed(20261027)
results=[]
with torch.inference_mode():
  for K,N in ((5120,34816),(5120,16384),(5120,14336),(6144,5120)):
    G=32
    pad=32 if (K//2)%1024==0 else 0
    packed=torch.randint(-(2**31),2**31-1,(N,K//8+pad),device="cuda",dtype=torch.int32)[:,:K//8]
    scales=torch.rand(N,K//G,device="cuda",dtype=torch.float16)*0.01+0.001
    zeros=torch.randint(-(2**31),2**31-1,(N//8,K//G),device="cuda",dtype=torch.int32)
    nk=torch.linspace(0,N-1,32,device="cuda").long()
    kk=torch.arange(K,device="cuda")
    shifts=(kk%8//2)*4+(kk%2)*16
    q=(packed[nk[:,None],kk[None,:]//8] >> shifts[None,:])&15
    z=(zeros[nk[:,None]//8,kk[None,:]//G] >> (4*(nk[:,None]%8)))&15
    weights=(q-z).half()*scales[nk[:,None],kk[None,:]//G]
    for m in (5,1535,1536,1600,2496,4097,8192,8193):
        x=rdna._pad_activation_rows(torch.randn(m,K,device="cuda",dtype=torch.float16))
        rdna._AMD_QWEN27B_PREFILL=False
        native=rdna.triton_w4a16_skinny_fmt_gemm(x,packed,scales,G,zp=zeros)
        rdna._AMD_QWEN27B_PREFILL=True
        candidate=rdna.triton_w4a16_skinny_fmt_gemm(x,packed,scales,G,zp=zeros)
        reference=(x.float()@weights.float().t()).half()
        err=(candidate[:,nk].float()-reference.float()).norm()/reference.float().norm()
        delta=(candidate.float()-native.float()).norm()/native.float().norm()
        exact=torch.equal(candidate,native)
        assert torch.isfinite(candidate).all() and err<0.0015 and delta<0.0001,(m,err,delta)
        if not 1536<=m<=8192:assert exact,(m,"fallback changed")
        row={"M":m,"K":K,"N":N,"group_size":G,"relative_l2_fp32_reference":float(err),
             "relative_l2_native":float(delta),"native_bitwise_equal":exact,"passed":True}
        results.append(row);print(json.dumps(row),flush=True)
        del x,native,candidate,reference
    del packed,scales,zeros,weights
    torch.cuda.empty_cache()
rdna._AMD_QWEN27B_PREFILL=False
torch.cuda.synchronize()
print("RESULT_JSON="+json.dumps({"passed":True,"rows":results}),flush=True)

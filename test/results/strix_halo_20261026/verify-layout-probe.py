"""Compare v0.32 skinny layout with K-major only at current verify shapes."""
import json, statistics
import torch
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as rdna
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
from vllm.utils.platform_utils import num_compute_units

torch.set_num_threads(1); torch.manual_seed(20261027)
rows=[]
def timing(fn):
    samples=[]
    for _ in range(12):
        b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        b.record();fn();e.record();e.synchronize();samples.append(b.elapsed_time(e))
    return samples
with torch.inference_mode():
    for K,N in [(17408,5120),(5120,34816)]:
        G=32
        q=torch.randint(0,16,(N,K),device="cuda",dtype=torch.int32)
        z=torch.randint(0,16,(N,K//G),device="cuda",dtype=torch.int32)
        scale=torch.rand(N,K//G,device="cuda",dtype=torch.float16)*0.01+0.001
        shifts=torch.arange(8,device="cuda",dtype=torch.int32)*4
        native_q=rdna.pack_skinny_int4(q)
        native_z=(z.view(N//8,8,K//G)<<shifts[None,:,None]).sum(1,dtype=torch.int32).contiguous()
        kq=(q.t().contiguous().reshape(K,N//8,8)<<shifts).sum(2,dtype=torch.int32).contiguous()
        ks,kz=scale.t().contiguous(),native_z.t().contiguous()
        ni=torch.linspace(0,N-1,64,device="cuda").long()
        w=(q[ni]-z[ni].repeat_interleave(G,dim=1)).half()*scale[ni].repeat_interleave(G,dim=1)
        del q,z
        for M in (1,3,5,9):
            x=torch.randn(M,K,device="cuda",dtype=torch.float16)
            native=lambda:rdna._rdna_hybrid_w4a16_apply_impl(x,native_q,scale,native_z,None,num_compute_units(),G)
            major=lambda:triton_w4a16_gemm(a=x,b_q=kq,scales=ks,group_size=G,zp_bias=0,qzeros=kz)
            out1,out2=native(),major();torch.cuda.synchronize()
            ref=(x.float()@w.float().t()).half()
            err=(out2[:,ni].float()-ref.float()).norm()/ref.float().norm()
            delta=(out2.float()-out1.float()).norm()/out1.float().norm()
            assert torch.isfinite(out2).all() and err<0.003 and delta<0.003,(M,K,N,err,delta)
            a,b,c=timing(native),timing(major),timing(native)
            row={"M":M,"K":K,"N":N,"native_before_ms":a,"kmajor_ms":b,"native_after_ms":c,
                 "speedup":statistics.median(a+c)/statistics.median(b),"oracle_relative_l2":float(err),
                 "native_relative_l2":float(delta),"passed":True}
            rows.append(row);print(json.dumps(row),flush=True)
            del x,out1,out2,ref
        del native_q,native_z,kq,ks,kz,scale,w,ni
        torch.cuda.empty_cache()
print("RESULT_JSON="+json.dumps({"scope":"Random asymmetric group32 weights; microbench only", "rows":rows}),flush=True)

import numpy as np, sys
tag=sys.argv[1]; n=int(sys.argv[2]); dim=int(sys.argv[3]); nq=int(sys.argv[4]); K=100
base=np.fromfile(f"/root/work/bench/{tag}_base.f32",dtype=np.float32).reshape(n,dim)
q=np.fromfile(f"/root/work/bench/{tag}_query.f32",dtype=np.float32).reshape(nq,dim)
bn=(base*base).sum(1)
out=np.zeros((nq,K),dtype=np.int32)
step=256
for s in range(0,nq,step):
    e=min(s+step,nq)
    d=bn[None,:]-2.0*(q[s:e]@base.T)
    idx=np.argpartition(d,K,axis=1)[:,:K]
    dv=np.take_along_axis(d,idx,axis=1)
    order=np.argsort(dv,axis=1)
    out[s:e]=np.take_along_axis(idx,order,axis=1)
np.save(f"/root/work/bench/{tag}_gt.npy",out.astype(np.int64))
print("gt done",tag)

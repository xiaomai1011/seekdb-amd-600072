import numpy as np, sys
tag=sys.argv[1]; out=sys.argv[2]; nq=int(sys.argv[3])
gt=np.load(f"/root/work/bench/{tag}_gt.npy")
ids=np.fromfile(out,dtype=np.uint32).reshape(nq,-1)
for k in (10,100):
    h=sum(len(set(ids[i,:k].tolist())&set(gt[i,:k].tolist())) for i in range(nq))
    print(f"recall@{k} = {h/(nq*k):.4f}", end="  ")
print()

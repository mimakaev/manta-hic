"""Compare small vs medium arch runs: per-dataset val Spearman (raw), between-channel (bm, cell-type-specific),
and 'combined' = mean(raw, bm) -- the shrink-study metric. Uses the last logged epoch of each <name>.pth."""
import torch, glob, os, math, sys

def load(d):
    r={}
    for f in sorted(glob.glob(f"{d}/*.pth")):
        ck=torch.load(f, map_location="cpu", weights_only=False)
        h=ck["config"].get("history",[])
        if h: r[os.path.basename(f)[:-4]]=h
    return r

def val(h, key):
    v=h[-1]["val"].get(key)
    return v if (v is not None and not (isinstance(v,float) and math.isnan(v))) else None

small=load("/workspace/arch_compare/small")
medium=load("/workspace/arch_compare/medium")
names=sorted(set(small)|set(medium))
ep_s = len(next(iter(small.values()))) if small else 0
ep_m = len(next(iter(medium.values()))) if medium else 0
print(f"small: {ep_s} epochs logged | medium: {ep_m} epochs logged\n")
hdr=f"{'dataset':22s} | {'raw spearman':^17s} | {'between-ch (bm)':^17s} | {'combined':^17s}"
print(hdr); print("-"*len(hdr))
print(f"{'':22s} | {'small':>7s} {'medium':>8s} | {'small':>7s} {'medium':>8s} | {'small':>7s} {'medium':>8s}")
for n in names:
    hs, hm = small.get(n), medium.get(n)
    def trio(h):
        if not h: return (None,None,None)
        r=val(h,"spearman"); b=val(h,"spearman_bm")
        c=(r+b)/2 if (r is not None and b is not None) else None
        return r,b,c
    rs,bs,cs=trio(hs); rm,bm,cm=trio(hm)
    f=lambda x: f"{x:.3f}" if isinstance(x,float) else "  -  "
    print(f"{n:22s} | {f(rs):>7s} {f(rm):>8s} | {f(bs):>7s} {f(bm):>8s} | {f(cs):>7s} {f(cm):>8s}")
# means over datasets where both present
def mean_combined(runs):
    vals=[]
    for n,h in runs.items():
        r=val(h,"spearman"); b=val(h,"spearman_bm")
        if r is not None:
            vals.append((r+b)/2 if b is not None else r)
    return sum(vals)/len(vals) if vals else float('nan')
print(f"\nmean raw spearman  small={sum(val(h,'spearman') for h in small.values())/len(small):.3f}"
      f"  medium={sum(val(h,'spearman') for h in medium.values())/len(medium):.3f}")

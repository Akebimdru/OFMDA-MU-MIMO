import numpy as np, json
import mapproxy
from mapproxy import Layout, mrc_snr, astar, resample
from cands import B, ROOM, BS, part
P=lambda cx,cy: dict(cx=cx,cy=cy,r=0.4,h=5.0,mat="metal")
starts=[(1.0,1.0),(14.25,3.0),(1.0,7.0),(14.25,5.0)]
goals=[(15.0,7.0),(1.0,7.0),(15.0,1.0),(1.0,1.0)]
base=part+[B(6.0,6.5,0.0,2.8,5,"concrete"), B(9.5,10.0,5.2,8.0,5,"concrete")]
gx=np.arange(.25,16,.5); gy=np.arange(.25,8,.5)
def plen(L):
    gb=L.footprint_blocked(gx,gy); out=[]
    free=np.argwhere(~gb)
    def cell(p):
        d=(gx[free[:,1]]-p[0])**2+(gy[free[:,0]]-p[1])**2; return tuple(free[np.argmin(d)])
    for s,g in zip(starts,goals):
        cp=astar(gb,cell(s),cell(g)); poly=[s]+[(gx[c[1]],gy[c[0]]) for c in cp]+[g]
        out.append(np.sum(np.linalg.norm(np.diff(np.array(poly),axis=0),axis=1)))
    return np.array(out)
L0=Layout(dict(name="",room=ROOM,bs=BS,starts=starts,goals=goals,boxes=base)); l0=plen(L0)
pts=np.array(starts)
# start SNR w/o pillars (averaged over 3x3 neighbourhood to reduce diffuse randomness)
def start_snr(L):
    off=np.array([(dx,dy) for dx in (-.25,0,.25) for dy in (-.25,0,.25)])
    return np.array([10*np.log10(np.mean(mrc_snr(L.channel(p+off)))) for p in pts])
s0=start_snr(L0)
res=[]
for x in np.arange(2.0,7.01,0.5):
    for y in np.arange(4.75,7.51,0.25):
        bx=base+[P(x,y),P(16-x,8-y)]
        L=Layout(dict(name="",room=ROOM,bs=BS,starts=starts,goals=goals,boxes=bx))
        # skip if pillar overlaps blocks/gates or starts
        if any(np.hypot(*(np.array(q)-[x,y]))<1.0 or np.hypot(*(np.array(q)-[16-x,8-y]))<1.0 for q in starts+goals): continue
        try: l=plen(L)
        except Exception: continue
        ss=start_snr(L)
        res.append(dict(x=x,y=y,dlen=np.round(l-l0,2).tolist(),n_hit=int(np.sum(l-l0>0.05)),sum_d=float(np.sum(l-l0)),snr_drop=np.round(s0-ss,1).tolist()))
res=[r for r in res if max(r["snr_drop"])<1.0 and r["n_hit"]>=2]
res.sort(key=lambda r:(-r["n_hit"],-r["sum_d"]))
print("base len",np.round(l0,2),"base start snr",np.round(s0,1))
for r in res[:15]: print(r)

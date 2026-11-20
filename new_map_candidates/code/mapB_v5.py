import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import mapproxy
from mapproxy import Layout, mrc_snr, run
from cands import B, ROOM, BS, part
P=lambda cx,cy,r=0.4,mat="metal": dict(cx=cx,cy=cy,r=r,h=5.0,mat=mat)
starts=[(1.0,1.0),(14.25,3.0),(1.0,7.0),(14.25,5.0)]
goals=[(15.0,7.0),(1.0,7.0),(15.0,1.0),(1.0,1.0)]
SPEC=dict(name="B v2: partition + 2 gates + metal pillars", room=ROOM, bs=BS, starts=starts, goals=goals,
  boxes=part+[B(6.0,6.5,0.0,2.8,5,"concrete"), B(9.5,10.0,5.2,8.0,5,"concrete"), P(7.0,5.5), P(9.0,2.5)])
out={}
for n in (None,-25):
    mapproxy.CSI["nmse_db"]=n
    r=run(SPEC,f"maps/proxyB5_{'perfect' if n is None else n}.png")
    if n is None: paths=r["paths"]["shortest"]; snr0=r
    out["perfect" if n is None else f"nmse{n}"]=r["shortest"]
out["start_snr_db"]=snr0["start_goal_snr_db"]; out["snr_median_db"]=snr0["snr_median_db"]
L=Layout(SPEC); rr=0.25; xs=np.arange(rr/2,16,rr); ys=np.arange(rr/2,8,rr); X,Y=np.meshgrid(xs,ys)
s=10*np.log10(np.mean(mrc_snr(L.channel(np.c_[X.ravel(),Y.ravel()])),-1)).reshape(X.shape)
occ=L.footprint_blocked(xs,ys,0.0)
gx=np.arange(.25,16,.5); gy=np.arange(.25,8,.5); gb=L.footprint_blocked(gx,gy)
fig,axs=plt.subplots(2,1,figsize=(11,12))
for ax,mode in zip(axs,("layout","snr")):
    if mode=="layout":
        ax.imshow(gb,origin="lower",extent=(0,16,0,8),cmap="Greys",alpha=.25)
        ax.set_title("Map B v5 layout (gray cells: blocked for AMR radius 0.25 m, 0.5 m grid)")
    else:
        im=ax.imshow(np.ma.masked_where(occ,s),origin="lower",extent=(0,16,0,8),vmin=28,vmax=58)
        plt.colorbar(im,ax=ax,fraction=.025,label="MRC SNR [dB] (proxy, not RT)")
        ax.set_title("SNR proxy + shortest paths (8-dir A*)")
    for b in SPEC["boxes"]:
        if "cx" in b:
            ax.add_patch(Circle((b["cx"],b["cy"]),b["r"],fc="lightsteelblue",ec="k")); ax.text(b["cx"],b["cy"]-0.7,"metal pillar\nr 0.4, h 5 m",ha="center",fontsize=7)
        else:
            ax.add_patch(Rectangle((b["x0"],b["y0"]),b["x1"]-b["x0"],b["y1"]-b["y0"],fc="dimgray",ec="k"))
            ax.text((b["x0"]+b["x1"])/2,(b["y0"]+b["y1"])/2,f"concrete {b['h']:g} m",ha="center",va="center",fontsize=7,color="w",rotation=90 if (b["y1"]-b["y0"])>2*(b["x1"]-b["x0"]) else 0)
    ax.plot(8,4,"r^",ms=13,mec="w",label="BS (8, 4, 5 m), 4-el ULA along x")
    for i,(a,g) in enumerate(zip(starts,goals)):
        ax.add_patch(Circle(a,0.25,fill=False,ec=f"C{i}",lw=2.2)); ax.plot(*g,"*",color=f"C{i}",ms=15,mec="k",mew=.5)
        ax.text(a[0],a[1]+0.4,f"AMR{i+1}",color=f"C{i}",ha="center",fontsize=9,fontweight="bold",bbox=dict(fc="w",alpha=.8,lw=0,pad=.6))
        if mode=="layout":
            ax.annotate("",g,a,arrowprops=dict(arrowstyle="->",color=f"C{i}",ls="--"))
        else:
            ax.plot(*paths[i].T,"-",color=f"C{i}",lw=2)
    ax.set_xlim(0,16); ax.set_ylim(0,8); ax.set_aspect("equal"); ax.set_xticks(range(17)); ax.set_yticks(range(9)); ax.grid(alpha=.3)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.06),fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig("maps/B_v5_preview.png",dpi=130,bbox_inches="tight")
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:v[kk] for kk in ("travel_s","rmin","median_inst_min","frac_time_hybrid_beats_mumimo","shadow_time_frac")}) for k,v in out.items()},indent=0))

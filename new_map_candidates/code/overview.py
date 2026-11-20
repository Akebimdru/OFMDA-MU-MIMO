import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import mapproxy
from mapproxy import Layout, mrc_snr, run
from cands import FINAL, SPECS
out={}
fig,axs=plt.subplots(3,1,figsize=(10,17))
for ax,(k,sp) in zip(axs,FINAL.items()):
    res={}
    for n in (None,-30,-25,-20):
        mapproxy.CSI["nmse_db"]=n
        r=run(sp,f"maps/proxy_{k}_{'perfect' if n is None else n}.png")
        if n is None: P=r["paths"]["shortest"]
        res["perfect" if n is None else f"nmse{n}"]={m:{kk:vv for kk,vv in r["shortest"].items()} for m in ["x"]}["x"]
    res["start_snr_db"]=r["start_goal_snr_db"]; res["snr_median_db"]=r["snr_median_db"]
    out[k]=res
    L=Layout(sp); rr=0.25; xs=np.arange(rr/2,16,rr); ys=np.arange(rr/2,8,rr); X,Y=np.meshgrid(xs,ys)
    s=10*np.log10(np.mean(mrc_snr(L.channel(np.c_[X.ravel(),Y.ravel()])),-1)).reshape(X.shape)
    occ=L.footprint_blocked(xs,ys,0.0)
    im=ax.imshow(np.ma.masked_where(occ,s),origin="lower",extent=(0,16,0,8),vmin=28,vmax=58,cmap="viridis")
    for b in sp["boxes"]:
        ax.add_patch(Rectangle((b["x0"],b["y0"]),b["x1"]-b["x0"],b["y1"]-b["y0"],fc="silver" if b["mat"]=="metal" else "dimgray",ec="k",lw=.7))
        ax.text((b["x0"]+b["x1"])/2,(b["y0"]+b["y1"])/2,f"{b['mat']} {b['h']:g} m",ha="center",va="center",fontsize=7,color="w" if b["mat"]!="metal" else "k",rotation=90 if (b["y1"]-b["y0"])>2*(b["x1"]-b["x0"]) else 0)
    ax.plot(*sp["bs"][:2],"r^",ms=13,mec="w",label=f"BS ({sp['bs'][0]:g}, {sp['bs'][1]:g}, 5 m), 4-el ULA along x")
    for i,(a,g) in enumerate(zip(sp["starts"],sp["goals"])):
        ax.add_patch(Circle(a,0.25,fill=False,ec=f"C{i}",lw=2.2)); ax.plot(*g,"*",color=f"C{i}",ms=14,mec="k",mew=.5)
        ax.plot(*P[i].T,"-",color=f"C{i}",lw=2,alpha=.9)
        ax.text(a[0],a[1]+0.38,f"AMR{i+1}",color=f"C{i}",ha="center",fontsize=9,fontweight="bold",bbox=dict(fc="w",alpha=.75,lw=0,pad=.6))
    p=res["perfect"]; q=res["nmse-25"]
    ax.set_title(f"{sp['name']}\nproxy, shortest paths: perfect CSI R_min  OFDMA {p['rmin']['ofdma_var']:.0f} / MU-MIMO {p['rmin']['mumimo']:.0f} / hybrid {p['rmin']['hybrid']:.0f} Mbps;"
                 f"  CSI NMSE -25 dB: {q['rmin']['ofdma_var']:.0f} / {q['rmin']['mumimo']:.0f} / {q['rmin']['hybrid']:.0f}",fontsize=10)
    ax.set_xlim(0,16); ax.set_ylim(0,8); ax.set_aspect("equal"); ax.set_xticks(range(17)); ax.set_yticks(range(9)); ax.grid(alpha=.25)
    ax.plot([],[],"k-",label="shortest path (8-dir A*, 0.5 m grid)"); ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.07),ncol=2,fontsize=8,frameon=False)
fig.colorbar(im,ax=axs,fraction=0.02,pad=0.02,label="MRC SNR [dB] (proxy: LoS + 1st-order reflections + diffuse, no RT)")
fig.savefig("maps/00_candidates_overview.png",dpi=130,bbox_inches="tight")
json.dump(out,open("maps/candidates_proxy_summary.json","w"),indent=1)
print(json.dumps({k:{m:v[m]["rmin"] for m in v if m.startswith(("perf","nmse"))} for k,v in out.items()},indent=0))

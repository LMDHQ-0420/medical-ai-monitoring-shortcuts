"""Fifteen title-free publication panels with varied academic graphic forms."""

from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, PathPatch, Rectangle, Patch
from matplotlib.path import Path as MplPath
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import result_data as rd
from style import COLORS, FONT, TASK_COLORS, clean_axis, figure, legend

TL = dict(zip(rd.TASKS, rd.TASK_LABELS))
TC = dict(zip(rd.TASKS, TASK_COLORS))
ML = dict(zip(rd.MODEL_KEYS, rd.MODEL_LABELS))
INK = COLORS["ink"]
ZERO = COLORS["gray_light"]
MODEL_COLORS = [COLORS[x] for x in ("blue", "coral", "teal", "orange", "purple")]
MC = dict(zip(rd.MODEL_KEYS, MODEL_COLORS))


def _seed_combo(frame, columns):
    return frame.groupby(rd.KEYS, sort=False)[columns].mean().reset_index()


def _summary(values):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    return np.quantile(v,[.10,.25,.50,.75,.90])


def _box_v(ax,x,values,color,width=.22):
    p10,q1,med,q3,p90=_summary(values)
    ax.plot([x,x],[p10,p90],color=INK,lw=.75,zorder=2)
    ax.plot([x-width*.28,x+width*.28],[p10,p10],color=INK,lw=.75)
    ax.plot([x-width*.28,x+width*.28],[p90,p90],color=INK,lw=.75)
    ax.add_patch(Rectangle((x-width/2,q1),width,q3-q1,fc=color,ec=COLORS["gray"],lw=.62,zorder=3))
    ax.plot([x-width/2,x+width/2],[med,med],color=INK,lw=1.35,zorder=4)


def _box_h(ax,y,values,color,height=.30):
    p10,q1,med,q3,p90=_summary(values)
    ax.plot([p10,p90],[y,y],color=INK,lw=.75,zorder=2)
    ax.plot([p10,p10],[y-height*.28,y+height*.28],color=INK,lw=.75)
    ax.plot([p90,p90],[y-height*.28,y+height*.28],color=INK,lw=.75)
    ax.add_patch(Rectangle((q1,y-height/2),q3-q1,height,fc=color,ec=COLORS["gray"],lw=.62,zorder=3))
    ax.plot([med,med],[y-height/2,y+height/2],color=INK,lw=1.35,zorder=4)


def _density(values,bins=72):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    lo,hi=v.min(),v.max(); pad=max((hi-lo)*.08,1e-4)
    h,e=np.histogram(v,bins=bins,range=(lo-pad,hi+pad),density=True)
    k=np.array([1,2,4,7,10,12,10,7,4,2,1],float); k/=k.sum()
    h=np.convolve(h,k,mode="same"); return (e[:-1]+e[1:])/2,h


def _violin_v(ax,x,values,color,width=.34,side="both",outline=False,summary="rules"):
    c,h=_density(values); d=h/h.max()*width
    if side in ("both","left"):
        ax.fill_betweenx(c,x-d,x,fc=_light(color,.22),ec="none",zorder=2)
        if outline: ax.plot(x-d,c,color=color,lw=.7,zorder=3)
    if side in ("both","right"):
        ax.fill_betweenx(c,x,x+d,fc=_light(color,.22),ec="none",zorder=2)
        if outline: ax.plot(x+d,c,color=color,lw=.7,zorder=3)
    q1,med,q3=np.quantile(values,[.25,.50,.75])
    reach=float(np.interp(med,c,d))
    if summary=="cross":
        if side=="left": span=(-reach,0); center=x-reach*.5
        elif side=="right": span=(0,reach); center=x+reach*.5
        else: span=(-reach,reach); center=x
        ax.plot([center,center],[q1,q3],color=COLORS["gray_dark"],lw=.82,zorder=5)
        ax.plot([x+span[0],x+span[1]],[med,med],color=INK,lw=1.18,zorder=6)
    else:
        def rule(q,color,lw,z):
            local=float(np.interp(q,c,d)); span=(-local,0) if side=="left" else (0,local) if side=="right" else (-local,local)
            ax.plot([x+span[0],x+span[1]],[q,q],color=color,lw=lw,zorder=z)
        for q in (q1,q3): rule(q,COLORS["gray"],.62,5)
        rule(med,INK,1.18,6)


def _violin_h(ax,y,values,color,width=.34,side="both"):
    c,h=_density(values); d=h/h.max()*width
    if side in ("both","down"):
        ax.fill_between(c,y-d,y,fc=_light(color,.22),ec="none",zorder=2)
    if side in ("both","up"):
        ax.fill_between(c,y,y+d,fc=_light(color,.22),ec="none",zorder=2)
    q1,med,q3=np.quantile(values,[.25,.50,.75])
    def rule(q,color,lw,z):
        reach=float(np.interp(q,c,d)); span=(-reach,0) if side=="down" else (0,reach) if side=="up" else (-reach,reach)
        ax.plot([q,q],[y+span[0],y+span[1]],color=color,lw=lw,zorder=z)
    for q in (q1,q3): rule(q,COLORS["gray"],.62,5)
    rule(med,INK,1.18,6)


def _ridge_h(ax,y,values,color,height=.58):
    """One-sided filled density with full-height quartile and median rules."""
    c,h=_density(values,80); d=h/h.max()*height
    ax.fill_between(c,y,y+d,fc=_light(color,.25),ec="none",zorder=2)
    ax.plot(c,y+d,color=color,lw=.75,zorder=3)
    q1,med,q3=np.quantile(values,[.25,.50,.75])
    for q in (q1,q3):
        local=float(np.interp(q,c,d)); ax.plot([q,q],[y,y+local],color=COLORS["gray"],lw=.58,zorder=4)
    local=float(np.interp(med,c,d)); ax.plot([med,med],[y,y+local],color=INK,lw=1.08,zorder=5)


def _sina_h(ax,y,values,color,height=.28):
    """Deterministic density-scaled circular jitter for a horizontal distribution."""
    v=np.asarray(values,float); v=v[np.isfinite(v)]; order=np.argsort(v); sv=v[order]
    c,h=_density(sv,80); local=np.interp(sv,c,h); local=local/max(local.max(),1e-12)
    phase=((np.arange(len(sv))*.61803398875)%1)-.5
    jitter=phase*2*height*local
    ax.plot(sv,y+jitter,marker="o",ls="",ms=1.45,mfc=_light(color,.12),mec="none",alpha=.78,zorder=2)
    q1,med,q3=np.quantile(sv,[.25,.50,.75])
    ax.plot([q1,q3],[y,y],color=COLORS["gray"],lw=.72,zorder=4)
    ax.plot([med,med],[y-height*.72,y+height*.72],color=INK,lw=1.05,zorder=5)


def _zero_if_all(axes,orientation="horizontal",lw=.65):
    """Draw zero guides only when every subplot in the panel contains zero."""
    limits=[ax.get_ylim() if orientation=="horizontal" else ax.get_xlim() for ax in axes]
    supported=all(lo < 0 < hi for lo,hi in limits)
    if supported:
        for ax in axes:
            if orientation=="horizontal": ax.axhline(0,color=ZERO,ls=(0,(3,2)),lw=lw,zorder=1)
            else: ax.axvline(0,color=ZERO,ls=(0,(3,2)),lw=lw,zorder=1)
    return supported


def _light(color,amount=.68):
    rgb=np.array(mcolors.to_rgb(color)); return tuple(rgb+(1-rgb)*amount)


def _alluvial(ax,x0,x1,y0,y1,width,color,alpha=.82):
    verts=[(x0,y0-width/2),(x0+(x1-x0)*.38,y0-width/2),(x0+(x1-x0)*.62,y1-width/2),(x1,y1-width/2),(x1,y1+width/2),(x0+(x1-x0)*.62,y1+width/2),(x0+(x1-x0)*.38,y0+width/2),(x0,y0+width/2),(x0,y0-width/2)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),transform=ax.transAxes,fc=color,ec=COLORS["white"],lw=.35,alpha=alpha))


def fig1a():
    fig=figure(80,64); ax=fig.add_axes([.02,.025,.96,.95]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Circle((.12,.79),.075,fc=_light(COLORS["blue"],.72),ec=INK,lw=1.0)); ax.add_patch(Circle((.12,.83),.022,fc=COLORS["white"],ec=INK,lw=.8)); ax.plot([.12,.12],[.808,.74],color=INK,lw=1.0); ax.plot([.085,.155],[.77,.77],color=INK,lw=.9)
    ax.text(.12,.66,"Same Patient",ha="center",fontsize=FONT["category"],weight="bold",color=INK)
    x=np.linspace(.27,.88,150); y=.82+.018*np.sin(np.linspace(0,8*np.pi,150)); y[66:79]+=.045*np.sin(np.linspace(0,np.pi,13)); ax.plot(x,y,color=COLORS["blue_dark"],lw=1.8)
    ax.text(.58,.91,"Disease Signal",ha="center",fontsize=FONT["milestone"],weight="bold",color=INK)
    times=np.array([.29,.37,.40,.52,.65,.68,.79,.86]); heights=np.array([.10,.06,.14,.08,.13,.05,.11,.07])
    for xx,hh in zip(times,heights): ax.plot([xx,xx],[.58,.58+hh],color=COLORS["coral"],lw=2.6,solid_capstyle="round")
    ax.plot([.27,.89],[.58,.58],color=INK,lw=.7); ax.text(.58,.48,"Observation Trace",ha="center",fontsize=FONT["milestone"],weight="bold",color=INK)
    for cx,label,fill in [(.34,"D",COLORS["blue_light"]),(.67,"D + O",COLORS["coral_light"])]: ax.add_patch(Ellipse((cx,.31),.25,.16,fc=_light(fill,.48),ec=INK,lw=.9)); ax.text(cx,.31,label,ha="center",va="center",fontsize=FONT["emphasis"],weight="bold",color=INK)
    ax.annotate("",xy=(.525,.31),xytext=(.475,.31),arrowprops=dict(arrowstyle="-|>",lw=.9,color=INK))
    ax.text(.50,.15,"ODI = AUROC(D + O) − AUROC(D)",ha="center",fontsize=FONT["milestone"],weight="bold",color=INK)
    return fig


def fig1b():
    d=rd.exp1(); fig=figure(190,80); gs=fig.add_gridspec(1,5,left=.07,right=.995,bottom=.20,top=.97,wspace=.32); axes=[]; xs=np.arange(len(rd.MODEL_KEYS))
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax); st=d[d.task==task]; data=[st[st.model==m].odi.to_numpy() for m in rd.MODEL_KEYS]
        bp=ax.boxplot(data,positions=xs,widths=.56,patch_artist=True,showfliers=False,whis=(10,90),medianprops=dict(color=INK,lw=1.0),whiskerprops=dict(color=INK,lw=.65),capprops=dict(color=INK,lw=.65))
        for box in bp["boxes"]: box.set(fc=_light(TC[task],.38),ec=COLORS["gray"],lw=.6)
        for mi,model in enumerate(rd.MODEL_KEYS):
            for wi,w in enumerate(rd.WINDOWS):
                med=np.median(st[(st.window==w)&(st.model==model)].odi); ax.plot(mi+(wi-2)*.075,med,marker="o",ls="",ms=2.7,mfc=TASK_COLORS[wi],mec=COLORS["white"],mew=.3,zorder=4)
        local=st.odi.to_numpy(); pad=max((local.max()-local.min())*.08,.004); ax.set_xlim(-.45,len(rd.MODEL_KEYS)-.55); ax.set_ylim(local.min()-pad,local.max()+pad); ax.set_xticks(xs,rd.MODEL_LABELS,rotation=35,ha="right")
        ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["category"],weight="bold",color=INK); clean_axis(ax)
    axes[0].set_ylabel("ODI",fontweight="bold"); fig.text(.53,.045,"Model",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig1c():
    d=rd.exp1(); fig=figure(110,64); ax=fig.add_axes([.11,.18,.88,.80])
    groups=[("disease",COLORS["blue"],"Disease Only"),("observation",COLORS["orange"],"Observation Only"),("mixed",COLORS["teal"],"Disease + Observation")]
    offsets=[-.22,0,.22]
    for x,task in enumerate(rd.TASKS):
        st=d[d.task==task]
        for off,(column,color,_) in zip(offsets,groups):
            _violin_v(ax,x+off,st[column].to_numpy(),color,.105,"both")
    values=d[[g[0] for g in groups]].to_numpy(); lo,hi=np.nanquantile(values,[.005,.995]); pad=max((hi-lo)*.05,.01)
    ax.set_xlim(-.52,4.52); ax.set_ylim(lo-pad,hi+pad); ax.set_xticks(range(5),rd.TASK_LABELS,rotation=25,ha="right"); ax.set_ylabel("AUROC",fontweight="bold",labelpad=2); ax.tick_params(axis="both",pad=1.5); clean_axis(ax)
    fig.text(.55,.012,"Outcome",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig2a():
    _,d=rd.exp2(); fig=figure(190,80); gs=fig.add_gridspec(1,5,left=.07,right=.995,bottom=.20,top=.97,wspace=.34); xs=np.arange(len(rd.MODEL_KEYS)); axes=[]
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax); st=d[d.task==task]; data=[st[st.model==m].odi.to_numpy() for m in rd.MODEL_KEYS]
        bp=ax.boxplot(data,positions=xs,widths=.56,patch_artist=True,showfliers=False,whis=(10,90),medianprops=dict(color=INK,lw=1.0),whiskerprops=dict(color=INK,lw=.65),capprops=dict(color=INK,lw=.65))
        for box in bp["boxes"]: box.set(fc=_light(TC[task],.38),ec=COLORS["gray"],lw=.6)
        for mi,model in enumerate(rd.MODEL_KEYS):
            for wi,w in enumerate(rd.WINDOWS):
                med=np.median(st[(st.model==model)&(st.window==w)].odi); ax.plot(mi+(wi-2)*.075,med,"o",ms=2.7,mfc=TASK_COLORS[wi],mec=COLORS["white"],mew=.3,zorder=4)
        bounds=[]
        for values in data:
            p10,_,_,_,p90=_summary(values); bounds.extend([p10,p90])
        pad=max((max(bounds)-min(bounds))*.10,.003)
        ax.set_xlim(-.45,len(rd.MODEL_KEYS)-.55); ax.set_ylim(min(bounds)-pad,max(bounds)+pad*2.6); ax.set_xticks(xs,rd.MODEL_LABELS,rotation=35,ha="right"); ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["category"],weight="bold",color=INK); clean_axis(ax)
    _zero_if_all(axes,"horizontal",.65)
    axes[0].set_ylabel("Diagnosis-Stratified ODI",fontweight="bold"); fig.text(.53,.045,"Model",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig2b():
    raw=rd.exp1()[rd.KEYS+["seed","odi"]]; psm,_=rd.exp2(); d=raw.merge(psm[rd.KEYS+["seed","odi"]],on=rd.KEYS+["seed"],suffixes=("_raw","_psm"))
    fig=figure(96,110); gs=fig.add_gridspec(5,1,left=.13,right=.98,bottom=.11,top=.98,hspace=.15); axes=[]; xs=np.arange(5)
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[i]); axes.append(ax); st=d[d.task==task]; local=[]
        for x,w in enumerate(rd.WINDOWS):
            vr=st[st.window==w].odi_raw.to_numpy(); vp=st[st.window==w].odi_psm.to_numpy(); local.extend(vr); local.extend(vp); _violin_v(ax,x,vr,COLORS["teal"],.34,"left",summary="cross"); _violin_v(ax,x,vp,COLORS["orange"],.34,"right",summary="cross")
        pad=max((max(local)-min(local))*.08,.003); ax.set_xlim(-.45,4.45); ax.set_ylim(min(local)-pad,max(local)+pad*1.7); ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["milestone"],weight="bold",color=INK); clean_axis(ax); ax.tick_params(axis="x",labelbottom=i==4,length=0)
    _zero_if_all(axes,"horizontal",.55)
    axes[-1].set_xticks(xs,[f"{w} h" for w in rd.WINDOWS]); fig.text(.55,.025,"Prediction Window",ha="center",fontsize=FONT["axis"],weight="bold",color=INK); fig.text(.015,.51,"ODI",rotation=90,ha="center",va="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig2c():
    _,s=rd.exp2(); diags=["cardiovascular","infection","other","renal","respiratory"]; dl=["Cardio","Infection","Other","Renal","Resp."]
    fig=figure(94,110); gs=fig.add_gridspec(5,1,left=.15,right=.98,bottom=.11,top=.98,hspace=.40); axes=[]
    for j,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[j]); axes.append(ax); st=s[s.task==task]; allv=st.odi.to_numpy(); lo=min(-.012,allv.min()-.006); hi=allv.max()+.012
        for i,diag in enumerate(diags):
            y=4-i; v=st[st.diagnosis==diag].odi.to_numpy(); _ridge_h(ax,y,v,TC[task],.54)
        ax.set_xlim(lo,hi); ax.set_ylim(-.5,5.05); ax.set_yticks(range(4,-1,-1),dl); ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["milestone"],weight="bold",color=INK); clean_axis(ax); ax.tick_params(axis="x",labelbottom=True,length=2)
    _zero_if_all(axes,"vertical",.6)
    fig.text(.54,.025,"ODI",ha="center",fontsize=FONT["axis"],weight="bold",color=INK); fig.text(.025,.53,"Diagnosis Stratum",rotation=90,ha="center",va="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig3c():
    raise RuntimeError('旧 exp3 完整模型差值无效；运行 fig/pnas_plots.py')
    d=rd.exp3(); fig=figure(84,58); ax=fig.add_axes([.20,.18,.77,.80]); ys=np.arange(5)[::-1]
    allv=d.mixed_change.dropna().to_numpy(); lo,hi=np.quantile(allv,[.01,.99]); lim=max(abs(lo),abs(hi),.0025)*1.10
    for y,task in zip(ys,rd.TASKS):
        values=d.loc[d.task==task,"mixed_change"].to_numpy(); _sina_h(ax,y,values,TC[task],.31)
    ax.axvline(0,color=ZERO,ls=(0,(3,2)),lw=.7); ax.set_xlim(-lim,lim); ax.set_ylim(-.18,4.78); ax.set_yticks(ys,rd.TASK_LABELS); ax.tick_params(axis="both",pad=1.4); clean_axis(ax)
    fig.text(.595,.012,"Full-Model AUROC Change",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig3a():
    d=rd.exp3(); fig=figure(106,58); ax=fig.add_axes([.12,.18,.87,.80]); xs=np.arange(5)
    for x,task in zip(xs,rd.TASKS):
        s=d[d.task==task]; _violin_v(ax,x,s.original.to_numpy(),COLORS["coral"],.34,"left",summary="cross"); _violin_v(ax,x,s.residual.to_numpy(),COLORS["blue"],.34,"right",summary="cross")
    vals=d[["original","residual"]].to_numpy(); lo,hi=np.nanquantile(vals,[.005,.995]); pad=max((hi-lo)*.06,.005)
    ax.axhline(0,color=ZERO,ls=(0,(3,2)),lw=.7); ax.set_xlim(-.48,4.48); ax.set_ylim(lo-pad,hi+pad); ax.set_xticks(xs,rd.TASK_LABELS,rotation=24,ha="right"); ax.set_ylabel("ODI",fontweight="bold",labelpad=2); ax.tick_params(axis="both",pad=1.4); clean_axis(ax)
    fig.text(.555,.012,"Outcome",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig3b():
    raise RuntimeError('Fig. 3B 必须由共享 F 的正式预测计算 reference_gain；运行 fig/pnas_plots.py')
    d=rd.exp3(); fig=figure(190,80); gs=fig.add_gridspec(1,5,left=.07,right=.995,bottom=.20,top=.97,wspace=.34); axes=[]; xs=np.arange(len(rd.MODEL_KEYS))
    for ti,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[0,ti]); axes.append(ax); st=d[d.task==task]; data=[st[st.model==m].mixed_change.to_numpy() for m in rd.MODEL_KEYS]
        bp=ax.boxplot(data,positions=xs,widths=.56,patch_artist=True,showfliers=False,medianprops=dict(color=INK,lw=1.0),whiskerprops=dict(color=INK,lw=.65),capprops=dict(color=INK,lw=.65))
        for box in bp["boxes"]: box.set(fc=_light(TC[task],.38),ec=COLORS["gray"],lw=.6)
        for mi,model in enumerate(rd.MODEL_KEYS):
            for wi,w in enumerate(rd.WINDOWS):
                med=np.median(st[(st.model==model)&(st.window==w)].mixed_change); ax.plot(mi+(wi-2)*.075,med,marker="o",ls="",ms=2.7,mfc=TASK_COLORS[wi],mec=COLORS["white"],mew=.3,zorder=4)
        bounds=[]
        for values in data:
            p10,_,_,_,p90=_summary(values); bounds.extend([p10,p90])
        span=max(bounds)-min(bounds); lower=max(span*.08,.002); upper=max(span*.34,.005); ax.set_xlim(-.45,len(rd.MODEL_KEYS)-.55); ax.set_ylim(min(bounds)-lower,max(bounds)+upper); ax.set_xticks(xs,rd.MODEL_LABELS,rotation=35,ha="right"); ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["category"],weight="bold",color=INK); clean_axis(ax)
    _zero_if_all(axes,"horizontal",.65)
    axes[0].set_ylabel("AUROC(D+T) − AUROC(D)",fontweight="bold"); fig.text(.53,.045,"Model",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig4a():
    a=rd.age(); ages=["<45","45-65","65-80",">80"]
    fig=figure(102,86); gs=fig.add_gridspec(5,1,left=.13,right=.99,bottom=.11,top=.99,hspace=.18); axes=[]; xs=np.arange(4)
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[i]); axes.append(ax); s=a[a.task==task]; local=[]
        for x,age in zip(xs,ages):
            vi=s[(s.age_group==age)&(s.source=="mimic_internal")].odi.dropna().to_numpy(); ve=s[(s.age_group==age)&(s.source=="eicu_external")].odi.dropna().to_numpy(); local.extend(vi); local.extend(ve)
            _violin_v(ax,x,vi,COLORS["blue"],.33,"left",summary="cross"); _violin_v(ax,x,ve,COLORS["coral"],.33,"right",summary="cross")
        lo,hi=np.quantile(local,[.005,.995]); span=hi-lo; lower=max(span*.06,.004); upper=max(span*.28,.010); ax.set_xlim(-.42,3.42); ax.set_ylim(lo-lower,hi+upper)
        show_x=i==4; ax.set_xticks(xs,rd.AGE_LABELS if show_x else []); ax.tick_params(axis="x",length=0,pad=1.3); ax.tick_params(axis="y",pad=1.3)
        ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["milestone"],weight="bold",color=INK); clean_axis(ax)
    _zero_if_all(axes,"horizontal",.5)
    fig.text(.555,.012,"Age Group",ha="center",fontsize=FONT["axis"],weight="bold",color=INK); fig.text(.018,.55,"ODI",rotation=90,ha="center",va="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig4b():
    g=rd.age_gaps(); fig=figure(88,86); gs=fig.add_gridspec(5,1,left=.13,right=.98,bottom=.11,top=.99,hspace=.18); axes=[]
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[i]); axes.append(ax); s=g[g.task==task]; allv=[]
        for y,col,color in [(1,"mimic_internal",COLORS["blue"]),(0,"eicu_external",COLORS["coral"])]:
            v=s[col].dropna().to_numpy(); allv.extend(v); _sina_h(ax,y,v,color,.27)
        lo,hi=np.quantile(allv,[.005,.995]); pad=max((hi-lo)*.06,.006); ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(-.38,1.38); ax.set_yticks([0,1],["eICU","MIMIC"])
        ax.text(.04,.95,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["milestone"],weight="bold",color=INK); ax.tick_params(axis="x",labelbottom=i==4,pad=1.3); ax.tick_params(axis="y",pad=1.2); clean_axis(ax)
    _zero_if_all(axes,"vertical",.55)
    fig.text(.56,.012,"Age-Gap ODI (>80 − ≤45)",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig4c():
    g=rd.age_gaps(); g["change"]=g.eicu_external-g.mimic_internal
    fig=figure(190,72); gs=fig.add_gridspec(1,5,left=.07,right=.995,bottom=.18,top=.97,wspace=.40); axes=[]; xs=np.arange(5)
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax); s=g[g.task==task]; local=[]
        for x,w in zip(xs,rd.WINDOWS):
            v=s[s.window==w].change.dropna().to_numpy(); local.extend(v); _box_v(ax,x,v,_light(TC[task],.32),.52)
            for mi,model in enumerate(rd.MODEL_KEYS):
                med=np.median(s[(s.window==w)&(s.model==model)].change.dropna()); ax.plot(x+(mi-(len(rd.MODEL_KEYS)-1)/2)*.065,med,marker="o",ls="",ms=2.4,mfc=MC[model],mec=COLORS["white"],mew=.3,zorder=7)
        lo,hi=np.quantile(local,[.005,.995]); pad=max((hi-lo)*.07,.008)
        ax.set_ylim(lo-pad,hi+pad); ax.set_xlim(-.45,4.45); ax.set_xticks(xs,[f"{w} h" for w in rd.WINDOWS],rotation=25,ha="right")
        ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["category"],weight="bold",color=INK); clean_axis(ax)
    _zero_if_all(axes,"horizontal",.65)
    axes[0].set_ylabel("External − Internal Age Gap",fontweight="bold"); fig.text(.53,.045,"Prediction Window",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig5a():
    h=rd.hospitals(); internal=h.groupby(rd.KEYS+["seed"],sort=False).odi_mimic.mean().reset_index(); external=h.groupby(rd.KEYS+["seed"],sort=False).odi_eicu.median().reset_index()
    fig=figure(84,84); ax=fig.add_axes([.15,.14,.84,.84]); xs=np.arange(5); allv=[]
    for x,task in zip(xs,rd.TASKS):
        vi=internal.loc[internal.task==task,"odi_mimic"].dropna().to_numpy(); ve=external.loc[external.task==task,"odi_eicu"].dropna().to_numpy(); allv.extend(vi); allv.extend(ve)
        _violin_v(ax,x,vi,COLORS["blue"],.34,"left",summary="cross"); _violin_v(ax,x,ve,COLORS["coral"],.34,"right",summary="cross")
    lo,hi=np.quantile(allv,[.005,.995]); pad=max((hi-lo)*.06,.008); ax.axhline(0,color=ZERO,ls=(0,(3,2)),lw=.65); ax.set_xlim(-.45,4.45); ax.set_ylim(lo-pad,hi+pad); ax.set_xticks(xs,rd.TASK_LABELS,rotation=28,ha="right"); ax.set_ylabel("ODI",fontweight="bold",labelpad=2); ax.tick_params(axis="both",pad=1.3); clean_axis(ax)
    fig.text(.57,.012,"Outcome",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig5b():
    h=rd.hospitals(); fig=figure(190,82); gs=fig.add_gridspec(1,5,left=.07,right=.995,bottom=.20,top=.97,wspace=.38); xs=np.arange(5); axes=[]
    for i,task in enumerate(rd.TASKS):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax); s=h[h.task==task]; local=[]
        for x,w in zip(xs,rd.WINDOWS):
            v=s[s.window==w].odi_shrinkage.dropna().to_numpy(); local.extend(v); _box_v(ax,x,v,_light(TC[task],.30),.52)
            for mi,model in enumerate(rd.MODEL_KEYS):
                med=np.median(s[(s.model==model)&(s.window==w)].odi_shrinkage.dropna()); ax.plot(x+(mi-(len(rd.MODEL_KEYS)-1)/2)*.065,med,marker="o",ls="",ms=2.6,mfc=MC[model],mec=COLORS["white"],mew=.3,zorder=7)
        lo,hi=np.quantile(local,[.01,.99]); span=hi-lo; lower=max(span*.06,.01); upper=max(span*.25,.025)
        ax.set_xlim(-.45,4.45); ax.set_ylim(lo-lower,hi+upper); ax.set_xticks(xs,[f"{w} h" for w in rd.WINDOWS],rotation=28,ha="right")
        ax.text(.04,.96,TL[task],transform=ax.transAxes,ha="left",va="top",fontsize=FONT["category"],weight="bold",color=INK); clean_axis(ax)
    _zero_if_all(axes,"horizontal",.65)
    axes[0].set_ylabel("Internal − External ODI",fontweight="bold"); fig.text(.53,.045,"Prediction Window",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def fig5c():
    h=rd.hospitals(); g=h.groupby(rd.KEYS+["hospital_id"],sort=False).mean(numeric_only=True).reset_index(); elig=g[(g.odi_mimic>0)&g.odi_eicu.notna()].copy(); elig["reversed"]=(elig.odi_eicu<0).astype(float)
    rates=elig.groupby(["task","hospital_id"],sort=False).reversed.mean().reset_index(); overall=elig.groupby(["task","model"],sort=False).reversed.mean().reset_index()
    fig=figure(106,84); ax=fig.add_axes([.17,.14,.81,.84]); ys=np.arange(5)[::-1]
    for y,task in zip(ys,rd.TASKS):
        v=rates.loc[rates.task==task,"reversed"].to_numpy(); c,dens=_density(v,70); ridge=dens/dens.max()*.60
        ax.fill_between(c,y,y+ridge,fc=_light(TC[task],.24),ec="none",zorder=2); ax.plot(c,y+ridge,color=TC[task],lw=.75,zorder=3)
        q1,med,q3=np.quantile(v,[.25,.50,.75])
        for q in (q1,q3):
            local=float(np.interp(q,c,ridge)); ax.plot([q,q],[y,y+local],color=COLORS["gray"],lw=.58,zorder=4)
        local=float(np.interp(med,c,ridge)); ax.plot([med,med],[y,y+local],color=INK,lw=1.05,zorder=5)
        model_rates=overall[overall.task==task].set_index("model").reversed
        for mi,model in enumerate(rd.MODEL_KEYS):
            ax.plot(model_rates[model],y-.12,marker="o",ls="",ms=2.8,mfc=MC[model],mec=COLORS["white"],mew=.35,zorder=6)
    ax.set_xlim(0,1); ax.set_ylim(-.30,4.72); ax.set_yticks(ys,rd.TASK_LABELS); ax.xaxis.set_major_formatter(PercentFormatter(1,0)); ax.tick_params(axis="both",pad=1.3); clean_axis(ax)
    fig.text(.58,.012,"Configurations Reversed within Hospital",ha="center",fontsize=FONT["axis"],weight="bold",color=INK)
    return fig


def _point_handles(labels, colors):
    return [Line2D([],[],marker="o",ls="",ms=5.2,mfc=c,mec=COLORS["white"],mew=.45,label=l)
            for l,c in zip(labels,colors)]


_fig5b_handles=_point_handles(rd.MODEL_LABELS,MODEL_COLORS)
_fig5b_handles=[_fig5b_handles[i] for i in (0,3,1,4,2)]


LEGEND_SPECS = {
    "fig1b": dict(handles=_point_handles([f"{w} h" for w in rd.WINDOWS],TASK_COLORS),ncol=5,width=58,height=8),
    "fig1c": dict(handles=[Patch(fc=_light(c,.22),ec="none",label=l) for _,c,l in [("d",COLORS["blue"],"Disease Only"),("o",COLORS["orange"],"Observation Only"),("m",COLORS["teal"],"Disease + Observation")]],ncol=3,width=88,height=8),
    "fig2a": dict(handles=_point_handles([f"{w} h" for w in rd.WINDOWS],TASK_COLORS),ncol=5,width=58,height=8),
    "fig2b": dict(handles=[Patch(fc=_light(COLORS["teal"],.22),ec="none",label="Original cohort"),Patch(fc=_light(COLORS["orange"],.22),ec="none",label="Matched cohort")],ncol=2,width=80,height=8),
    "fig3a": dict(handles=[Patch(fc=_light(COLORS["coral"],.22),ec="none",label="Original ODI"),Patch(fc=_light(COLORS["blue"],.22),ec="none",label="Residual ODI")],ncol=2,width=62,height=8),
    "fig3b": dict(handles=_point_handles([f"{w} h" for w in rd.WINDOWS],TASK_COLORS),ncol=5,width=58,height=8),
    "fig4a": dict(handles=[Patch(fc=_light(COLORS["blue"],.22),ec="none",label="MIMIC Internal"),Patch(fc=_light(COLORS["coral"],.22),ec="none",label="eICU External")],ncol=2,width=62,height=8),
    "fig4c": dict(handles=_point_handles(rd.MODEL_LABELS,MODEL_COLORS),ncol=len(rd.MODEL_KEYS),width=92,height=8),
    "fig5a": dict(handles=[Patch(fc=_light(COLORS["blue"],.22),ec="none",label="MIMIC Internal"),Patch(fc=_light(COLORS["coral"],.22),ec="none",label="eICU Hospital Median")],ncol=2,width=76,height=8),
    "fig5b": dict(handles=_fig5b_handles,ncol=3,width=62,height=13),
    "fig5c": dict(handles=_point_handles(rd.MODEL_LABELS,MODEL_COLORS),ncol=len(rd.MODEL_KEYS),width=92,height=8),
}


BUILDERS={"fig1a":fig1a,"fig1b":fig1b,"fig1c":fig1c,"fig2a":fig2a,"fig2b":fig2b,"fig2c":fig2c,"fig3a":fig3a,"fig3b":fig3b,"fig3c":fig3c,"fig4a":fig4a,"fig4b":fig4b,"fig4c":fig4c,"fig5a":fig5a,"fig5b":fig5b,"fig5c":fig5c}

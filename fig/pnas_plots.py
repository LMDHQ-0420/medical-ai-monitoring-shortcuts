"""从显式正式统计文件生成 PNAS 主图；拒绝缺格点或历史结果。"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
TASKS=['icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation']
LABELS=['ICU mortality','Hospital mortality','AKI','Deterioration','Escalation']
MODELS=['lr','rf','xgb','lgbm','mlp'];MODEL_LABELS=['LR','RF','XGBoost','LightGBM','MLP']
COLORS=['#3070A8','#40A58E','#D68A36','#9E67A7','#C35D59']

def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                      'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def save(fig,out,name):
 fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=400,bbox_inches='tight');plt.close(fig)

def label(ax,letter):ax.text(-.08,1.08,letter,transform=ax.transAxes,fontsize=12,fontweight='bold')

def plot1(main,out):
 fig,axes=plt.subplots(1,3,figsize=(15.2,3.7),gridspec_kw={'width_ratios':[1.1,1.45,1.45]})
 fig.subplots_adjust(wspace=.66)
 ax=axes[0];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 ax.add_patch(Circle((.13,.53),.08,facecolor='#dceaf5',edgecolor='#3070A8'))
 ax.plot([.07,.10,.12,.14,.16,.19],[.53,.53,.60,.46,.53,.53],color='#3070A8',lw=1.4)
 for x,y,letter,title,color in [(.45,.75,'O','Observation','#40A58E'),(.45,.31,'T','Treatment','#D68A36'),(.83,.53,'F','Full model','#9E67A7')]:
  ax.add_patch(Circle((x,y),.08,facecolor=color,alpha=.2,edgecolor=color))
  ax.text(x,y,letter,ha='center',va='center',fontsize=10,fontweight='bold')
  ax.text(x,y-.13,title,ha='center',va='top',fontsize=8)
 ax.text(.13,.40,'Clinical state (D)',ha='center',fontsize=8)
 for start,end in [((.22,.53),(.73,.53)),((.53,.72),(.75,.58)),((.53,.34),(.75,.48))]:
  ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'#666666','lw':.8})
 ax.text(.5,.08,'ODI = AUROC(F) − AUROC(D)',ha='center',fontsize=8)
 label(ax,'A')
 config=main.groupby(['task','model','window_h'],as_index=False).odi_raw.mean()
 mat=config.groupby(['task','model']).odi_raw.mean().unstack().reindex(index=TASKS,columns=MODELS)
 ax=axes[1];im=ax.imshow(mat.to_numpy(),cmap='YlGnBu',aspect='auto',vmin=0,vmax=mat.max().max())
 ax.set_xticks(range(5),MODEL_LABELS,rotation=25,ha='right');ax.set_yticks(range(5),LABELS)
 ax.set_title('Mean raw ODI',fontsize=9);fig.colorbar(im,ax=ax,fraction=.04,pad=.04);label(ax,'B')
 ax=axes[2]
 for i,model in enumerate(MODELS):
  g=config[config.model==model].groupby('window_h').odi_raw.mean().reindex([3,6,12,18,24])
  ax.plot(g.index,g.values,marker='o',ms=3.2,lw=1.4,label=MODEL_LABELS[i],color=COLORS[i])
 ax.axhline(0,color='#888888',lw=.7);ax.set_xticks([3,6,12,18,24]);ax.set_xlabel('Prediction window (h)');ax.set_ylabel('Mean raw ODI');ax.legend(frameon=False,ncol=2,fontsize=7);label(ax,'C')
 save(fig,out,'fig1')

def plot2(balance, matched, context, out):
    """A 为匹配平衡，B 为固定模型测试集增益，C 为入院情境分层。"""
    required = {'task','model','window_h','seed','odi_raw_original','odi_raw_matched',
                'odi_residual_original','odi_residual_matched'}
    if len(matched) != 625 or not required.issubset(matched) or context.empty or balance.empty:
        raise ValueError('Fig. 2 正式匹配、平衡或入院情境结果缺失')
    fig = plt.figure(figsize=(9.0, 7.2))
    grid = fig.add_gridspec(2, 2, left=.18, right=.96, bottom=.15, top=.91,
                            width_ratios=[1.0, 1.25], hspace=.60, wspace=.52)
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1])]
    ax = axes[0]
    b = (balance.assign(abs_smd=lambda d:d.smd.abs())
         .groupby(['covariate','stage']).abs_smd.median().unstack().sort_values('before'))
    y = np.arange(len(b))
    ax.plot(b.before,y,'o',ms=3.5,label='Before matching',color='#3070A8')
    ax.plot(b.after,y,'s',ms=3.5,label='After matching',color='#D68A36')
    for i,(before,after) in enumerate(zip(b.before,b.after)):
        ax.plot([before,after],[i,i],color='#cccccc',lw=.7,zorder=0)
    ax.axvline(.1,color='#888888',ls='--',lw=.8)
    ax.set_yticks(y,b.index,fontsize=7)
    ax.set_xlabel('Median absolute standardized difference')
    ax.legend(frameon=False,fontsize=7,loc='lower right')
    label(ax,'A')
    ax = axes[1];x = np.arange(5)
    columns = ['odi_raw_original','odi_raw_matched','odi_residual_original','odi_residual_matched']
    g = matched.groupby('task')[columns].mean().reindex(TASKS)
    for col,offset,color,marker,name in [
        ('odi_raw_original',-.18,'#3070A8','o','Raw, original cohort'),
        ('odi_raw_matched',-.06,'#D68A36','s','Raw, matched cohort'),
        ('odi_residual_original',.06,'#40A58E','o','Treatment-adjusted, original'),
        ('odi_residual_matched',.18,'#9E67A7','s','Treatment-adjusted, matched')]:
        ax.scatter(x+offset,g[col],color=color,marker=marker,s=26,label=name)
    ax.axhline(0,color='#888888',lw=.7)
    ax.set_xticks(x,LABELS,rotation=32,ha='right',fontsize=7)
    ax.set_ylabel('Mean AUROC gain')
    ax.legend(frameon=False,fontsize=6.5,ncol=1,loc='upper left')
    label(ax,'B')
    ax = axes[2]
    order = ['Emergency','Transfer','Elective/procedure','Other/unknown']
    d = context.groupby(['task','stratum']).odi_residual.mean().unstack().reindex(index=TASKS,columns=order)
    lim = max(abs(float(d.min().min())),abs(float(d.max().max())),.01)
    im = ax.imshow(d.to_numpy(),aspect='auto',cmap='BrBG',vmin=-lim,vmax=lim)
    ax.set_yticks(range(5),LABELS,fontsize=7)
    ax.set_xticks(range(4),order,rotation=36,ha='right',fontsize=7)
    fig.colorbar(im,ax=ax,fraction=.05,pad=.02,label='Treatment-adjusted AUROC gain')
    label(ax,'C')
    save(fig,out,'fig2')

def plot_eicu_diagnosis(diagnosis, out):
    """另报 eICU 截点前已录入诊断的探索性残余增益。"""
    required = {'task','diagnosis','odi_residual','status'}
    if not required.issubset(diagnosis) or diagnosis.empty:
        raise ValueError('eICU 截点前诊断分层输入缺失')
    order = ['infection','cardiovascular','respiratory','renal','other_recorded','not_recorded']
    d = (diagnosis.loc[diagnosis.status.eq('complete')]
         .groupby(['task','diagnosis']).odi_residual.mean().unstack()
         .reindex(index=TASKS,columns=order))
    fig,ax = plt.subplots(figsize=(8.5,4.2))
    lim=max(abs(float(d.min().min())),abs(float(d.max().max())),.01)
    image=ax.imshow(d.to_numpy(),aspect='auto',cmap='BrBG',vmin=-lim,vmax=lim)
    ax.set_yticks(range(5),LABELS)
    ax.set_xticks(range(len(order)),['Infection','Cardiovascular','Respiratory','Renal','Other recorded','Not recorded'],rotation=30,ha='right')
    fig.colorbar(image,ax=ax,fraction=.035,pad=.03,label='Treatment-adjusted AUROC gain')
    ax.set_title('eICU diagnoses entered before prediction cutoff',fontsize=10)
    fig.subplots_adjust(left=.20,right=.87,bottom=.31,top=.88)
    fig.text(.5,.035,'Exploratory external comparison; diagnosis categories may overlap',ha='center',fontsize=8)
    save(fig,out,'extended_eicu_diagnosis')

def plot3(main,out):
 config=main.groupby(['task','model','window_h'],as_index=False)[['odi_raw','odi_residual','reference_gain']].mean()
 fig,axes=plt.subplots(1,3,figsize=(13.4,3.7))
 for ax,col,letter,title in zip(axes,['odi_raw','reference_gain','odi_residual'],'ABC',
        ['Raw observation contrast','Reference-model gain','Treatment-adjusted contrast']):
  data=[config.loc[config.task==task,col].dropna().to_numpy() for task in TASKS]
  ax.boxplot(data,positions=range(5),showfliers=False,patch_artist=True,
             boxprops={'facecolor':'#BBD4E6','edgecolor':'#427AA3'},medianprops={'color':'#183A50'},
             whiskerprops={'color':'#427AA3'},capprops={'color':'#427AA3'})
  ax.axhline(0,color='#888888',lw=.7);ax.set_xticks(range(5),LABELS,rotation=35,ha='right',fontsize=7)
  ax.set_title(title,fontsize=9);ax.set_ylabel('AUROC difference');label(ax,letter)
 save(fig,out,'fig3')

def plot4(age,out):
 a=age[age.dataset=='mimic_iv'];ext=age[age.dataset=='eicu'];order=['<45','45-64','65-80','>80']
 fig,axes=plt.subplots(1,3,figsize=(13.4,3.7))
 ax=axes[0]
 for i,model in enumerate(MODELS):
  g=a[a.model==model].groupby('age_group').odi_raw.mean().reindex(order)
  ax.plot(range(4),g.values,marker='o',ms=3,lw=1.3,color=COLORS[i],label=MODEL_LABELS[i])
 ax.set_xticks(range(4),order);ax.set_ylabel('Mean raw ODI');ax.set_xlabel('Age group (years)');ax.legend(frameon=False,ncol=2,fontsize=7);label(ax,'A')
 keys=['task','model','window_h','seed'];pivot=a.pivot_table(index=keys,columns='age_group',values='odi_raw').dropna(subset=['<45','>80']);pivot['gap']=pivot['>80']-pivot['<45']
 ax=axes[1];groups=[pivot.loc[pivot.index.get_level_values('model')==m,'gap'].to_numpy() for m in MODELS]
 ax.boxplot(groups,showfliers=False);ax.axhline(0,color='#888888',lw=.7);ax.set_xticks(range(1,6),MODEL_LABELS,rotation=25,ha='right');ax.set_ylabel('ODI(>80) − ODI(<45)');label(ax,'B')
 if len(ext):
  pext=ext.pivot_table(index=keys,columns='age_group',values='odi_raw').dropna(subset=['<45','>80']);pext['gap']=pext['>80']-pext['<45']
  joined=pivot[['gap']].join(pext[['gap']],lsuffix='_internal',rsuffix='_external',how='inner')
  ax=axes[2];gap=joined.gap_external-joined.gap_internal
  ax.hist(gap,bins=24,color='#40A58E',alpha=.7);ax.axvline(0,color='#333333',lw=.8);ax.set_xlabel('External − internal age gap');ax.set_ylabel('Configurations × seeds');label(ax,'C')
 save(fig,out,'fig4')

def plot5(external,out):
 fig,axes=plt.subplots(1,3,figsize=(13.4,3.7))
 e=external.copy();e=e[np.isfinite(e.odi_raw)&np.isfinite(e.internal_odi_raw)]
 ax=axes[0];ax.scatter(e.internal_odi_raw,e.odi_raw,s=4,alpha=.15,color='#3070A8');xlo=float(e.internal_odi_raw.min());xhi=float(e.internal_odi_raw.max());pad=.04*(xhi-xlo)
 ax.plot([xlo,xhi],[xlo,xhi],color='#555555',lw=.8);ax.set_xlim(xlo-pad,xhi+pad);ax.set_xlabel('Internal raw ODI');ax.set_ylabel('External hospital raw ODI');label(ax,'A')
 ax=axes[1];data=[e[e.task==task].odi_shrinkage.to_numpy() for task in TASKS];ax.boxplot(data,showfliers=False);ax.axhline(0,color='#888888',lw=.7);ax.set_xticks(range(1,6),LABELS,rotation=35,ha='right',fontsize=7);ax.set_ylabel('Internal − external ODI');label(ax,'B')
 ax=axes[2];rates=[]
 for task in TASKS:
  x=e[(e.task==task)&(e.internal_odi_raw>0)];rates.append((x.odi_raw<0).mean() if len(x) else np.nan)
 ax.bar(range(5),rates,color=COLORS);ax.set_ylim(0,1);ax.set_xticks(range(5),LABELS,rotation=35,ha='right',fontsize=7);ax.set_ylabel('External hospital reversal proportion');label(ax,'C')
 save(fig,out,'fig5')

def plot_extended(seq,out):
 fig,axes=plt.subplots(1,2,figsize=(9.2,3.5))
 means=seq.groupby(['task','window_h'],as_index=False)[['odi_raw','odi_residual','reference_gain']].mean()
 ax=axes[0]
 for i,task in enumerate(TASKS):
  g=means[means.task==task].sort_values('window_h')
  ax.plot(g.window_h,g.odi_raw,marker='o',ms=3,color=COLORS[i],label=LABELS[i])
 ax.axhline(0,color='#888888',lw=.7);ax.set_xticks([3,6,12,18,24]);ax.set_xlabel('Genuine hourly time steps');ax.set_ylabel('GRU-seq raw ODI');ax.legend(frameon=False,fontsize=7,ncol=2);label(ax,'A')
 ax=axes[1];x=np.arange(5);width=.28
 values=means.groupby('task')[['odi_residual','reference_gain']].mean().reindex(TASKS)
 ax.bar(x-width/2,values.odi_residual,width,label='Treatment-adjusted ODI',color='#3070A8')
 ax.bar(x+width/2,values.reference_gain,width,label='Reference-model gain',color='#D68A36')
 ax.axhline(0,color='#888888',lw=.7);ax.set_xticks(x,LABELS,rotation=30,ha='right',fontsize=7);ax.set_ylabel('AUROC difference');ax.legend(frameon=False,fontsize=7);label(ax,'B')
 save(fig,out,'extended_gru_seq')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();out=run/'figures';out.mkdir(exist_ok=True);style()
 main=pd.read_csv(run/'main_seed_metrics.csv.gz')
 if len(main)!=625 or main.groupby(['task','model','window_h']).seed.nunique().ne(5).any():raise ValueError('主实验 625 个正式格点未完成，拒绝绘制正式主图')
 age=pd.read_csv(run/'age_seed_metrics.csv.gz');external=pd.read_csv(run/'external_seed_metrics.csv.gz')
 if age.groupby(['task','model','window_h','seed','dataset']).size().unstack(fill_value=0).shape[0]!=625 or not len(external):
  raise ValueError('年龄和外部正式结果未覆盖全部格点')
 plot1(main,out);plot3(main,out);plot4(age,out);plot5(external,out)
 seq=pd.read_csv(run/'gru_seq_seed_metrics.csv.gz')
 if len(seq)==125:plot_extended(seq,out)
 matched=pd.read_csv(run/'matched_seed_metrics.csv.gz')
 context_path=run/'context_seed_metrics.csv.gz'
 balance_path=run/'fig2_matching/balance_diagnostics.csv'
 if matched.empty or not context_path.exists() or not balance_path.exists():
  (out/'fig2_status.json').write_text(json.dumps({'status':'incomplete','reason':'test-only clinical-state matching or admission-context analysis not complete'},indent=2)+'\n')
 else:
  plot2(pd.read_csv(balance_path),matched,pd.read_csv(context_path),out)
  (out/'fig2_status.json').write_text(json.dumps({'status':'complete','analysis':'frozen models evaluated on test-only matched pairs; eICU diagnosis reported separately'},indent=2)+'\n')
  plot_eicu_diagnosis(pd.read_csv(run/'diagnosis_seed_metrics.csv.gz'),out)
 print('Rendered fig1, fig3, fig4, fig5 from complete formal inputs; Fig2 status explicit')
if __name__=='__main__':main()

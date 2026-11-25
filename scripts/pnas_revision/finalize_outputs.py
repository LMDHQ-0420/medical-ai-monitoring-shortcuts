"""核对完整统计，填充论文数字、数据字典与图件来源状态。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd


def write_json(path,obj):
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 main=pd.read_csv(run/'main_seed_metrics.csv.gz');seq=pd.read_csv(run/'gru_seq_seed_metrics.csv.gz')
 ext=pd.read_csv(run/'external_seed_metrics.csv.gz');gext=pd.read_csv(run/'gru_seq_external_seed_metrics.csv.gz')
 age=pd.read_csv(run/'age_seed_metrics.csv.gz');gage=pd.read_csv(run/'gru_seq_age_seed_metrics.csv.gz')
 imain=pd.read_csv(run/'main_paired_intervals.csv.gz');iseq=pd.read_csv(run/'gru_seq_paired_intervals.csv.gz')
 missing={name:len(pd.read_csv(run/name)) for name in ['missing_grid.csv','gru_seq_missing_grid.csv','subgroup_missing_grid.csv','gru_seq_subgroup_missing_grid.csv']}
 if (len(main),len(seq),len(imain),len(iseq))!=(625,125,625,125) or any(missing.values()):
  raise ValueError(f'正式统计不完整：主表 {len(main)}、GRU {len(seq)}、区间 {len(imain)}/{len(iseq)}、缺失 {missing}')
 if not (age.groupby(['task','model','window_h','seed','dataset']).size().unstack(fill_value=0).shape[0]==625 and
         gage.groupby(['task','window_h','seed','dataset']).size().unstack(fill_value=0).shape[0]==125):
  raise ValueError('年龄分层未覆盖全部正式格点')
 numbers=json.loads((run/'manuscript_numbers.json').read_text())
 numbers['tabular_odi_residual_positive_seed_cells']=int((main.odi_residual>0).sum())
 numbers['gru_seq_odi_raw_positive_seed_cells']=int((seq.odi_raw>0).sum())
 numbers['gru_seq_odi_residual_positive_seed_cells']=int((seq.odi_residual>0).sum())
 numbers['gru_seq_configuration_cells_complete_5_seeds']=int(seq.groupby(['task','window_h']).seed.nunique().eq(5).sum())
 numbers['tabular_task_means']={str(k):{'odi_raw':float(v.odi_raw.mean()),'odi_residual':float(v.odi_residual.mean()),'reference_gain':float(v.reference_gain.mean()),'seed_cells':len(v)} for k,v in main.groupby('task')}
 numbers['gru_seq_task_means']={str(k):{'odi_raw':float(v.odi_raw.mean()),'odi_residual':float(v.odi_residual.mean()),'reference_gain':float(v.reference_gain.mean()),'seed_cells':len(v)} for k,v in seq.groupby('task')}
 numbers['external_task_summary']={}
 for task,group in ext.groupby('task'):
  eligible=group[group.internal_odi_raw>0]
  numbers['external_task_summary'][task]={'eligible_hospital_comparisons':len(group),'unique_hospitals':int(group.hospital_id.nunique()),
   'reversal_numerator':int((eligible.odi_raw<0).sum()),'reversal_denominator':len(eligible),
   'mean_internal_minus_external_odi':float(group.odi_shrinkage.mean())}
 numbers['subgroup_coverage']={'tabular_age_rows':len(age),'tabular_external_hospital_rows':len(ext),'gru_seq_age_rows':len(gage),
  'gru_seq_external_hospital_rows':len(gext),'missing_cells':missing}
 numbers['paired_interval_rows']={'tabular':len(imain),'gru_seq':len(iseq)}
 numbers['statistical_warning']='Task, window, seed, model, and hospital comparisons reuse patients; these rows are not independent people.'
 write_json(run/'manuscript_numbers.json',numbers)
 dictionary=json.loads((run/'data_dictionary.json').read_text())
 dictionary['interval_algorithm']='500 outcome-stratified patient-level bootstrap resamples per seed cell; D, DT and F use the same resampled patient indices; two-sided 2.5th and 97.5th percentile limits of paired AUROC differences.'
 dictionary['interval_files']={'tabular':'main_paired_intervals.csv.gz','gru_seq':'gru_seq_paired_intervals.csv.gz'}
 dictionary['interval_unit']='individual patient within a seed cell; intervals do not treat repeated tasks, windows, models, seeds or hospitals as independent patients'
 dictionary['external_reversal_denominator']='eligible hospital × task × model × window × seed comparisons with internal ODI > 0; hospital requires at least 20 patients and both outcome classes'
 dictionary['missing_reason_files']=list(missing)
 write_json(run/'data_dictionary.json',dictionary)
 mapping=run/'figure_data_map.csv';fig=pd.read_csv(mapping)
 fig.loc[fig.status=='waiting_full_grid','status']='rendered_formal_20261125'
 fig.loc[fig.status=='waiting_sequence_grid','status']='rendered_formal_20261125'
 fig.to_csv(mapping,index=False)
 lines=['# 最终正式统计核对','',f'运行标识：`{run.name}`。主分析 625/625 个种子格点、125/125 个五种子配置；GRU-seq 125/125 个种子格点、25/25 个五种子配置。患者级配对区间分别覆盖 {len(imain)} 与 {len(iseq)} 个格点，每格 500 次按结局分层重采样。','',
  f'主分析原始 ODI 为正：{int((main.odi_raw>0).sum())}/625 个种子格点，{numbers["main_odi_raw_positive_configurations_5_seeds"]}/125 个五种子配置。GRU-seq 为正：{int((seq.odi_raw>0).sum())}/125。格点与配置均不是独立患者。','',
  '| 任务 | 五模型平均原始 ODI | GRU-seq 平均原始 ODI | 外部 reversal 分子/分母 |','|---|---:|---:|---:|']
 for task in main.task.unique():
  a=numbers['tabular_task_means'][task];b=numbers['gru_seq_task_means'][task];e=numbers['external_task_summary'][task]
  lines.append(f'| {task} | {a["odi_raw"]:.6f} | {b["odi_raw"]:.6f} | {e["reversal_numerator"]}/{e["reversal_denominator"]} |')
 lines+=['','上述平均值先在每个 task×model×window 的五种子格点上取值，再对配置求算术平均；每任务配置平衡，因此与直接对该任务种子格点均值相同。外部 reversal 的有效单位为医院×配置×种子且内部 ODI>0。','',
  '跨库任务标签、截点风险人群和检验结果可用时间仍未统一核定，外部图件只作为探索性描述；匹配和诊断分层尚未正式运行。']
 (run/'final_statistics_report_zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'main':len(main),'gru_seq':len(seq),'intervals':[len(imain),len(iseq)],'age_rows':[len(age),len(gage)],'external_rows':[len(ext),len(gext)]},ensure_ascii=False))

if __name__=='__main__':main()

"""年龄分层的统一边界及配对 AUROC 差值。"""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.exp1.trainer import GROUPS


def age_label(age):
    return pd.cut(age,bins=[0,45,65,81,121],labels=['<45','45-64','65-80','>80'],right=False)


def contrasts(frame):
    if len(frame)<20 or frame.y_true.nunique()<2:return None
    auc={g:roc_auc_score(frame.y_true,frame[f'pred_{g}']) for g in GROUPS}
    return {'n':len(frame),'positive':int(frame.y_true.sum()),**{f'auc_{g}':float(x) for g,x in auc.items()},
            'odi_raw':float(auc['F']-auc['D']),'odi_residual':float(auc['F']-auc['DT']),
            'reference_gain':float(auc['DT']-auc['D'])}

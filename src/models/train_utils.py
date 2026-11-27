"""五种表格模型的统一训练入口。"""
from __future__ import annotations
import numpy as np
from .model_factory import create_model
EVAL_SET_STYLE=("xgb","lgbm")
TABULAR_MODELS=("lr","rf","xgb","lgbm","mlp")

def create_and_fit(model_type:str,config:dict,X_train:np.ndarray,y_train:np.ndarray,
                   X_val:np.ndarray,y_val:np.ndarray,logger=None):
    if model_type not in TABULAR_MODELS:
        raise ValueError(f"{model_type!r} 不属于五种表格模型")
    if X_train.ndim!=2 or X_val.ndim!=2:
        raise ValueError('表格模型输入必须为二维')
    model=create_model(model_type,config)
    if model_type in EVAL_SET_STYLE:model.fit(X_train,y_train,X_val=X_val,y_val=y_val)
    else:model.fit(X_train,y_train)
    return model

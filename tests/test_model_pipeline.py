"""验证五模型名单及年龄、外部评估的默认汇总流程。"""
from contextlib import nullcontext
import pandas as pd
from src.exp1.trainer import MODELS
from src.models.train_utils import TABULAR_MODELS
from src.exp5 import run_external


def test_active_model_set():
    assert MODELS == TABULAR_MODELS == ('lr', 'rf', 'xgb', 'lgbm', 'mlp')


def test_external_default_grid_completes_with_five_models(tmp_path, monkeypatch):
    monkeypatch.setattr(run_external, 'TASKS', ('icu_mortality',))
    monkeypatch.setattr(run_external, 'WINDOWS', (3,))
    monkeypatch.setattr(run_external, 'SEEDS', (42,))
    monkeypatch.setattr(run_external, 'experiment_log', lambda *a, **kw: nullcontext())
    calls=[]

    def evaluate(root, task, model, window, seed):
        calls.append(model)
        cell=root/'models/tabular'/task/model/f'seed{seed}'/f'{window}h'
        cell.mkdir(parents=True)
        # 格式测试数据，仅写入 pytest 临时目录。
        pd.DataFrame([{'age_group':'<45','odi_raw':0.01}]).to_csv(cell/'age_metrics.csv',index=False)
        pd.DataFrame([{'hospital_id':1,'odi_raw':0.02}]).to_csv(cell/'hospital_metrics.csv',index=False)

    monkeypatch.setattr(run_external, 'evaluate_tabular', evaluate)
    run_external.run_experiment(tmp_path,None,None,None,None)
    assert calls == list(MODELS)
    for exp in ('exp4','exp5'):
        table=pd.read_csv(tmp_path/exp/'results.csv.gz')
        assert len(table)==5 and set(table.model)==set(MODELS)
    assert not list((tmp_path/'models').rglob('*metrics.csv'))

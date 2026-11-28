"""患者划分固定登记：保留旧归属，新患者按 ID 与种子哈希分配。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.formal.identity import canonical_hash

SEEDS = (42, 43, 44, 45, 46)
SPLIT_VERSION = "legacy-preserving-hash-v1"


def make_registry(legacy_path: Path, current_path: Path, output: Path) -> pd.DataFrame:
    legacy = pd.read_parquet(legacy_path, columns=["subject_id"])
    current = pd.read_parquet(current_path, columns=["subject_id"])
    old_ids = np.sort(legacy.subject_id.dropna().unique())
    new_ids = np.sort(current.subject_id.dropna().unique())
    old_set = set(old_ids)
    records = []
    for seed in SEEDS:
        perm = old_ids.copy()
        np.random.default_rng(seed).shuffle(perm)
        n_train, n_val = int(len(perm) * .7), int(len(perm) * .1)
        original = {pid: ("train" if i < n_train else "validation" if i < n_train + n_val else "test")
                    for i, pid in enumerate(perm)}
        for pid in new_ids:
            if pid in old_set:
                split, source = original[pid], "legacy"
            else:
                digest = hashlib.sha256(f"{SPLIT_VERSION}|{seed}|{pid}".encode()).digest()
                value = int.from_bytes(digest[:8], "big") / 2**64
                split = "train" if value < .7 else "validation" if value < .8 else "test"
                source = "new_hash"
            records.append({"subject_id": pid, "seed": seed, "split": split, "source": source})
    result = pd.DataFrame(records)
    assert not result.duplicated(["subject_id", "seed"]).any()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    result.to_parquet(temp, index=False)
    temp.replace(output)
    return result


def split_frame(frame: pd.DataFrame, registry: pd.DataFrame, seed: int):
    mapping = registry.loc[registry.seed.eq(seed), ["subject_id", "split"]]
    tagged = frame.merge(mapping, on="subject_id", how="left", validate="many_to_one")
    if tagged.split.isna().any():
        raise ValueError("split registry 缺少患者")
    parts = tuple(tagged.loc[tagged.split.eq(key)].drop(columns="split").copy()
                  for key in ("train", "validation", "test"))
    ids = [set(part.subject_id) for part in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("患者跨 split")
    return parts


def split_id(registry: pd.DataFrame, seed: int) -> str:
    rows = registry.loc[registry.seed.eq(seed), ["subject_id", "split"]].sort_values("subject_id")
    return canonical_hash(rows.to_dict("records"))


def audit_legacy_conflicts(legacy_root: Path, registry: pd.DataFrame,
                           tasks: tuple[str, ...], windows: tuple[int, ...],
                           output: Path) -> pd.DataFrame:
    """量化旧版各任务独立洗牌和新版统一患者登记的归属冲突。"""
    records = []
    for window in windows:
        old = pd.read_parquet(legacy_root / f'mimic_iv_{window}h.parquet',
                              columns=['subject_id', *tasks])
        for task in tasks:
            ids = np.sort(old.loc[old[task].notna(), 'subject_id'].dropna().unique())
            for seed in SEEDS:
                shuffled = ids.copy()
                np.random.default_rng(seed).shuffle(shuffled)
                n_train = int(len(ids) * .7)
                n_val = int(len(ids) * .1)
                original = pd.DataFrame({'subject_id': shuffled,
                                         'legacy_split': ['train'] * n_train +
                                         ['validation'] * n_val +
                                         ['test'] * (len(ids) - n_train - n_val)})
                fixed = registry.loc[registry.seed.eq(seed), ['subject_id', 'split']]
                joined = original.merge(fixed, on='subject_id', how='left', validate='one_to_one')
                comparable = joined.split.notna()
                conflict = joined.loc[comparable, 'legacy_split'].ne(joined.loc[comparable, 'split'])
                records.append({'window_h': window, 'task': task, 'seed': seed,
                                'n_original': len(ids), 'n_comparable': int(comparable.sum()),
                                'n_conflict': int(conflict.sum()),
                                'conflict_fraction': float(conflict.mean()) if len(conflict) else np.nan})
    result = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result

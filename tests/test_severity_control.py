import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from src.exp2.severity_control import propensity_score_matching


class MatchingTests(unittest.TestCase):
    def match(self, df):
        scores = df["severity"].to_numpy()
        with patch("src.exp2.severity_control.LogisticRegression") as model:
            model.return_value.predict_proba.return_value = np.column_stack([1 - scores, scores])
            return propensity_score_matching(df, "high", ["severity"], caliper=0, seed=43)

    def test_failed_high_patient_is_excluded_and_later_success_retained(self):
        high = pd.DataFrame({"high": [1] * 4, "severity": [0.2, 0.3, 0.4, 0.5]},
                            index=[10, 20, 30, 40])
        failed = high.sample(frac=1, random_state=43).index[0]
        successful = high.drop(index=failed)
        low = successful.assign(high=0).set_axis([50, 60, 70])
        result = self.match(pd.concat([high, low]))
        self.assertCountEqual(result.loc[result.high == 1, "severity"], successful.severity)
        self.assertCountEqual(result.loc[result.high == 0, "severity"], successful.severity)
        self.assertEqual(result.attrs["psm_diagnostics"]["n_unmatched_high"], 1)
        self.assertEqual(result.attrs["psm_diagnostics"]["n_pairs"], 3)

    def test_all_successful_without_control_reuse(self):
        df = pd.DataFrame({"high": [1, 1, 0, 0], "severity": [0.2, 0.4, 0.2, 0.4]})
        result = self.match(df)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.attrs["psm_diagnostics"]["n_unmatched_high"], 0)

    def test_no_matches(self):
        df = pd.DataFrame({"high": [1, 1, 0, 0], "severity": [0.8, 0.9, 0.1, 0.2]})
        self.assertTrue(self.match(df).empty)

    def test_sorted_matching_agrees_with_full_scan_including_ties(self):
        rng = np.random.default_rng(123)
        for seed in (42, 43):
            for caliper in (0, 0.1, 1):
                scores = rng.choice([0.1, 0.2, 0.3, 0.4, 0.6, 0.8], size=100)
                df = pd.DataFrame({"high": [1] * 50 + [0] * 50,
                                   "severity": scores, "patient": range(100)},
                                  index=rng.permutation(np.arange(100) + 1000))
                high = df[df.high == 1].sample(frac=1, random_state=seed)
                low = df[df.high == 0]
                selected_high, selected_low, used = [], [], set()
                for idx, row in high.iterrows():
                    differences = (low.severity - row.severity).abs()
                    differences = differences[~low.index.isin(used)]
                    if differences.empty:
                        continue
                    nearest = differences.idxmin()
                    if differences[nearest] <= caliper * df.severity.std():
                        selected_high.append(idx)
                        selected_low.append(nearest)
                        used.add(nearest)
                with patch("src.exp2.severity_control.LogisticRegression") as model:
                    model.return_value.predict_proba.return_value = np.column_stack([1 - scores, scores])
                    actual = propensity_score_matching(df, "high", ["severity"], caliper, seed)
                expected = df.loc[selected_high + selected_low, "patient"].tolist()
                self.assertEqual(actual.patient.tolist(), expected)

    def test_psm_only_overwrites_and_preserves_other_analyses(self):
        spec = importlib.util.spec_from_file_location("severity_entry", CODE_DIR / "scripts/exp2_ablation_severity.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        df = pd.DataFrame({"subject_id": range(100), "icu_mortality": [0, 1] * 50})
        matched = df.copy()
        matched.attrs["psm_diagnostics"] = {"seed": 43}
        config = {"tasks": ["icu_mortality"], "time_windows": [6],
                  "training": {"seed": 42, "train_val_test_split": [0.7, 0.1, 0.2]}}
        original = {"psm": {"odi": -99}, "stratification": {"infection": 0.2},
                    "sofa_adjusted": {"auroc": 0.7}}
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path("config.json").write_text(json.dumps(config))
                Path("data/processed").mkdir(parents=True)
                Path("data/processed/mimic_iv_6h.parquet").touch()
                output = Path("results/exp2/icu_mortality/mixed/lr/seed43/6h_severity_control.json")
                output.parent.mkdir(parents=True)
                output.write_text(json.dumps(original))
                with patch.object(sys, "argv", ["exp2", "--config", "config.json", "--model", "lr", "--seed", "43", "--psm-only"]), \
                     patch.object(entry.pd, "read_parquet", return_value=df), \
                     patch.object(entry, "compute_oii", return_value=(pd.Series(np.arange(100), index=df.index), None)), \
                     patch.object(entry, "propensity_score_matching", return_value=matched) as matching, \
                     patch.object(entry, "_compute_odi_on_split", return_value={"odi": 0.1}), \
                     patch.object(entry, "stratify_by_diagnosis") as stratification, \
                     patch.object(entry, "sofa_adjusted_analysis") as sofa:
                    entry.main()
                    self.assertEqual(matching.call_args.kwargs["seed"], 43)
                    stratification.assert_not_called()
                    sofa.assert_not_called()
                actual = json.loads(output.read_text())
                self.assertEqual(actual["psm"]["odi"], 0.1)
                self.assertEqual(actual["stratification"], original["stratification"])
                self.assertEqual(actual["sofa_adjusted"], original["sofa_adjusted"])
                self.assertEqual(sum(actual["psm"]["diagnostics"]["split_rows"].values()), 100)
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()

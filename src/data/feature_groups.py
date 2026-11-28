"""正式预测输入组；身份和审计字段绝不进入模型输入。"""

CLINICAL_STATE_FEATURES = [
    "hr_first", "sbp_first", "dbp_first", "temp_first", "rr_first", "spo2_first",
    "creatinine_first", "lactate_first", "wbc_first", "hgb_first",
    "sodium_first", "potassium_first", "bilirubin_first", "inr_first",
    "age", "gender", "admission_source",
]
OBSERVATION_FEATURES = [
    "vitals_count_3h", "vitals_count_6h", "vitals_count_12h", "vitals_count_18h", "vitals_count_24h",
    "labs_count_3h", "labs_count_6h", "labs_count_12h", "labs_count_18h", "labs_count_24h",
    "creatinine_repeat_3h", "creatinine_repeat_6h", "creatinine_repeat_12h",
    "creatinine_repeat_18h", "creatinine_repeat_24h",
    "lactate_repeat_3h", "lactate_repeat_6h", "lactate_repeat_12h",
    "lactate_repeat_18h", "lactate_repeat_24h",
    "abg_count_24h", "has_lactate_6h", "has_creatinine_6h", "has_blood_culture_24h",
    "has_abg_24h", "has_troponin_24h", "orders_count_6h", "orders_count_24h",
    "has_cxr_24h", "cxr_count_24h",
]
TREATMENT_FEATURES = ["antibiotics_24h", "vasopressors_24h", "mechanical_ventilation_24h"]
LATEST_VITAL_FEATURES = ["hr_latest", "sbp_latest", "dbp_latest", "temp_latest", "rr_latest", "spo2_latest"]
PREDICTOR_SETS = {
    "clinical_state": CLINICAL_STATE_FEATURES,
    "observation": OBSERVATION_FEATURES,
    "clinical_state_treatment": CLINICAL_STATE_FEATURES + TREATMENT_FEATURES,
    "full_model": CLINICAL_STATE_FEATURES + OBSERVATION_FEATURES + TREATMENT_FEATURES,
}
DISPLAY_NAMES = {
    "clinical_state": "Clinical State",
    "observation": "Observation",
    "clinical_state_treatment": "Clinical State + Treatment",
    "full_model": "Full Model",
}
LABEL_COLS = ["icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation"]
TASK_DISPLAY_NAMES = {
    "icu_mortality": "ICU Death", "inhosp_mortality": "Hospital Death",
    "aki": "AKI", "sepsis_deterioration": "Deterioration", "escalation": "Escalation",
}


def features_for(predictor_set: str, state_representation: str = "first_measurements") -> list[str]:
    if predictor_set not in PREDICTOR_SETS:
        raise ValueError(f"未知输入组: {predictor_set}")
    if state_representation == "first_measurements":
        return PREDICTOR_SETS[predictor_set].copy()
    if state_representation != "first_plus_latest":
        raise ValueError(f"未知状态表征: {state_representation}")
    if predictor_set not in ("clinical_state_treatment", "full_model"):
        raise ValueError("状态增强仅比较治疗参考组和完整模型")
    cols = PREDICTOR_SETS[predictor_set].copy()
    return cols[:len(CLINICAL_STATE_FEATURES)] + LATEST_VITAL_FEATURES + cols[len(CLINICAL_STATE_FEATURES):]


def get_obs_features_for_window(window_h: int) -> list[str]:
    """匹配强度计算可选的单窗口观察列。"""
    return [column for column in OBSERVATION_FEATURES if column.endswith(f"_{window_h}h")]

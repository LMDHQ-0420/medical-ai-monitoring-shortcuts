"""Central registry of variable names for the three feature groups.

All modules import column names from here — never hardcode elsewhere.
"""

DISEASE_FEATURES: list[str] = [
    # Initial vital sign values (first recorded within window)
    "hr_first", "sbp_first", "dbp_first", "temp_first", "rr_first", "spo2_first",
    # Initial lab values (first recorded within window)
    "creatinine_first", "lactate_first", "wbc_first", "hgb_first",
    "sodium_first", "potassium_first", "bilirubin_first", "inr_first",
    # 入科时已知的静态人口学与入院来源。
    "age", "gender", "admission_source",
]

OBSERVATION_FEATURES: list[str] = [
    # Vital sign recording density (counts within time window)
    "vitals_count_3h", "vitals_count_6h", "vitals_count_12h", "vitals_count_18h", "vitals_count_24h",
    # Total lab orders
    "labs_count_3h", "labs_count_6h", "labs_count_12h", "labs_count_18h", "labs_count_24h",
    # Repeat testing counts
    "creatinine_repeat_3h", "creatinine_repeat_6h", "creatinine_repeat_12h",
    "creatinine_repeat_18h", "creatinine_repeat_24h",
    "lactate_repeat_3h", "lactate_repeat_6h", "lactate_repeat_12h",
    "lactate_repeat_18h", "lactate_repeat_24h",
    "abg_count_24h",
    # Key test availability (binary)
    "has_lactate_6h", "has_creatinine_6h", "has_blood_culture_24h",
    "has_abg_24h", "has_troponin_24h",
    # Order density
    "orders_count_6h", "orders_count_24h",
    # Imaging
    "has_cxr_24h", "cxr_count_24h",
]

MIXED_FEATURES: list[str] = [
    "antibiotics_24h",
    "vasopressors_24h",
    "mechanical_ventilation_24h",
]

ALL_FEATURES: list[str] = DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES

LABEL_COLS: list[str] = [
    "icu_mortality",
    "inhosp_mortality",
    "aki",
    "sepsis_deterioration",
    "escalation",
]

# Window-specific observation feature subsets (derived automatically)
def get_obs_features_for_window(window_h: int) -> list[str]:
    """Return observation features relevant to the given time window."""
    return [f for f in OBSERVATION_FEATURES if f.endswith(f"_{window_h}h")]

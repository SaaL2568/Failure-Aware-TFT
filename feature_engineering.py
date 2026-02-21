import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import timedelta
from config import Config

VITAL_THRESHOLDS = {
    "heart_rate":        {"low": 40,  "high": 130},
    "sbp":               {"low": 80,  "high": 180},
    "dbp":               {"low": 50,  "high": 110},
    "resp_rate":         {"low": 8,   "high": 30},
    "spo2":              {"low": 88,  "high": 100},
    "temperature":       {"low": 35,  "high": 39.5},
    "map":               {"low": 60,  "high": 110},
}

VITAL_ITEMID_MAP = {
    220045: "heart_rate",
    220179: "sbp",
    220180: "dbp",
    220210: "resp_rate",
    220277: "spo2",
    223761: "temperature",
    220050: "art_sbp",
    220051: "art_dbp",
    220052: "map",
}

LAB_ITEMID_MAP = {
    50912:  "creatinine",
    50971:  "potassium",
    50983:  "sodium",
    51006:  "bun",
    50902:  "chloride",
    50882:  "bicarbonate",
    50893:  "calcium",
    51221:  "hematocrit",
    51222:  "hemoglobin",
    51265:  "platelets",
    50813:  "lactate",
    50820:  "ph",
    50821:  "pao2",
    50818:  "paco2",
}

def _computeSlope(times_hours, values):
    if len(values) < 2:
        return np.nan
    try:
        slope, _, _, _, _ = linregress(times_hours, values)
        return slope
    except Exception:
        return np.nan

def _computeWindowFeatures(series_df, col, window_start, window_end):
    """
    Given a time-indexed series DataFrame and a column name,
    compute summary statistics over [window_start, window_end).
    Returns a dict of features.
    """
    mask = (series_df["charttime"] >= window_start) & (series_df["charttime"] < window_end)
    window = series_df.loc[mask, ["charttime", col]].dropna(subset=[col])

    prefix = col
    if len(window) == 0:
        return {
            f"{prefix}_mean":        np.nan,
            f"{prefix}_min":         np.nan,
            f"{prefix}_max":         np.nan,
            f"{prefix}_std":         np.nan,
            f"{prefix}_slope":       np.nan,
            f"{prefix}_last":        np.nan,
            f"{prefix}_delta":       np.nan,
            f"{prefix}_missing":     1.0,
            f"{prefix}_count":       0,
            f"{prefix}_above_high":  np.nan,
            f"{prefix}_below_low":   np.nan,
        }

    values = window[col].values
    times_hours = (
        (window["charttime"] - window_start).dt.total_seconds() / 3600
    ).values

    slope = _computeSlope(times_hours, values)
    delta = float(values[-1] - values[0]) if len(values) > 1 else np.nan

    thresholds = VITAL_THRESHOLDS.get(col, None)
    above_high = float(np.mean(values > thresholds["high"])) if thresholds else np.nan
    below_low = float(np.mean(values < thresholds["low"])) if thresholds else np.nan

    return {
        f"{prefix}_mean":       float(np.mean(values)),
        f"{prefix}_min":        float(np.min(values)),
        f"{prefix}_max":        float(np.max(values)),
        f"{prefix}_std":        float(np.std(values)) if len(values) > 1 else 0.0,
        f"{prefix}_slope":      slope,
        f"{prefix}_last":       float(values[-1]),
        f"{prefix}_delta":      delta,
        f"{prefix}_missing":    0.0,
        f"{prefix}_count":      len(values),
        f"{prefix}_above_high": above_high,
        f"{prefix}_below_low":  below_low,
    }

class FeatureEngineer:
    def __init__(self, icu_path, hosp_path):
        self.icu_path = icu_path
        self.hosp_path = hosp_path
        self.window_duration = timedelta(minutes=Config.OBSERVATION_WINDOW_MINUTES)

    def _loadCharteventsForStays(self, stay_ids):
        relevant_itemids = list(VITAL_ITEMID_MAP.keys())
        chunks = []
        for chunk in pd.read_csv(
            f"{self.icu_path}chartevents.csv",
            usecols=["stay_id", "charttime", "itemid", "valuenum", "warning"],
            parse_dates=["charttime"],
            chunksize=500_000,
        ):
            chunk = chunk[
                chunk["stay_id"].isin(stay_ids)
                & chunk["itemid"].isin(relevant_itemids)
                & (chunk["warning"] == 0)
            ]
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df["vital_name"] = df["itemid"].map(VITAL_ITEMID_MAP)
        return df

    def _loadLabeventsForStays(self, subject_ids):
        relevant_itemids = list(LAB_ITEMID_MAP.keys())
        chunks = []
        for chunk in pd.read_csv(
            f"{self.hosp_path}labevents.csv",
            usecols=["subject_id", "charttime", "itemid", "valuenum"],
            parse_dates=["charttime"],
            chunksize=500_000,
        ):
            chunk = chunk[
                chunk["subject_id"].isin(subject_ids)
                & chunk["itemid"].isin(relevant_itemids)
            ]
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df["lab_name"] = df["itemid"].map(LAB_ITEMID_MAP)
        return df

    def _pivotVitals(self, chartevents):
        pivoted = chartevents.pivot_table(
            index=["stay_id", "charttime"],
            columns="vital_name",
            values="valuenum",
            aggfunc="mean",
        ).reset_index()
        pivoted.columns.name = None
        return pivoted

    def _pivotLabs(self, labevents, cohort):
        labevents = labevents.merge(
            cohort[["subject_id", "stay_id"]], on="subject_id", how="left"
        )
        pivoted = labevents.pivot_table(
            index=["stay_id", "charttime"],
            columns="lab_name",
            values="valuenum",
            aggfunc="mean",
        ).reset_index()
        pivoted.columns.name = None
        return pivoted

    def computeFeaturesForWindow(self, vitals_pivoted, labs_pivoted, stay_id, window_end):
        window_start = window_end - self.window_duration

        stay_vitals = vitals_pivoted[vitals_pivoted["stay_id"] == stay_id].copy()
        stay_labs = labs_pivoted[labs_pivoted["stay_id"] == stay_id].copy()

        features = {}

        vital_cols = [c for c in stay_vitals.columns if c not in ("stay_id", "charttime")]
        for col in vital_cols:
            features.update(
                _computeWindowFeatures(stay_vitals, col, window_start, window_end)
            )

        lab_cols = [c for c in stay_labs.columns if c not in ("stay_id", "charttime")]
        for col in lab_cols:
            features.update(
                _computeWindowFeatures(stay_labs, col, window_start, window_end)
            )

        features["time_since_admission_hours"] = (
            window_end - stay_vitals["charttime"].min()
        ).total_seconds() / 3600 if len(stay_vitals) > 0 else np.nan

        return features

    def buildFeatureMatrix(self, cohort, prediction_times):
        """
        Build a full feature matrix for all (stay_id, prediction_time) pairs.

        Returns:
            DataFrame where each row is a prediction point with engineered features
        """
        stay_ids = set(cohort["stay_id"].tolist())
        subject_ids = set(cohort["subject_id"].tolist())

        print("Loading chart events...")
        chartevents = self._loadCharteventsForStays(stay_ids)
        vitals_pivoted = self._pivotVitals(chartevents)

        print("Loading lab events...")
        labevents = self._loadLabeventsForStays(subject_ids)
        labs_pivoted = self._pivotLabs(labevents, cohort)

        print(f"Computing features for {len(prediction_times):,} prediction points...")
        rows = []
        grouped = prediction_times.groupby("stay_id")

        for stay_id, group in grouped:
            stay_vitals = vitals_pivoted[vitals_pivoted["stay_id"] == stay_id]
            stay_labs = labs_pivoted[labs_pivoted["stay_id"] == stay_id]

            for _, pt_row in group.iterrows():
                window_end = pt_row["prediction_time"]
                features = self.computeFeaturesForWindow(
                    stay_vitals, stay_labs, stay_id, window_end
                )
                features["stay_id"] = stay_id
                features["prediction_time"] = window_end
                rows.append(features)

        feature_matrix = pd.DataFrame(rows)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix

    def addStaticFeatures(self, feature_matrix, cohort, static_df):
        """
        Merge in static features (age, gender, admission type etc) that
        don't change over time for a given stay.
        """
        merged = feature_matrix.merge(
            static_df, on="stay_id", how="left"
        )
        return merged

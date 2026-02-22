import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import timedelta
from config import Config


class MIMICPreprocessor:
    def __init__(self, data_loader):
        self.data_loader    = data_loader
        self.scalers        = {}
        self.label_encoders = {}

    # ------------------------------------------------------------------ #
    #  Cohort  —  stratified sampling to guarantee positive rate
    # ------------------------------------------------------------------ #
    def create_cohort(self):
        patients   = self.data_loader.load_patients()
        admissions = self.data_loader.load_admissions()
        icustays   = self.data_loader.load_icustays()

        cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        cohort = cohort.merge(patients, on="subject_id", how="inner")
        cohort = cohort[cohort["los"] >= Config.LOOKBACK_WINDOW_HOURS / 24].reset_index(drop=True)

        # Tag in-ICU deaths for stratification
        def _is_positive(row):
            if row.get("hospital_expire_flag", 0) == 1:
                return 1
            dt = row.get("deathtime", None)
            if pd.notna(dt) and row["intime"] <= dt <= row["outtime"]:
                return 1
            return 0

        cohort["_is_positive"] = cohort.apply(_is_positive, axis=1)

        if hasattr(Config, "SAMPLE_SIZE") and Config.SAMPLE_SIZE and len(cohort) > Config.SAMPLE_SIZE:
            pos   = cohort[cohort["_is_positive"] == 1]
            neg   = cohort[cohort["_is_positive"] == 0]
            n_pos = int(Config.SAMPLE_SIZE * len(pos) / len(cohort))
            n_neg = Config.SAMPLE_SIZE - n_pos
            n_pos = min(n_pos, len(pos))
            n_neg = min(n_neg, len(neg))
            cohort = pd.concat([
                pos.sample(n=n_pos, random_state=Config.SEED),
                neg.sample(n=n_neg, random_state=Config.SEED),
            ]).sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
            print(f"Stratified sample: {len(cohort)} stays "
                  f"({n_pos} positive / {n_neg} negative)")

        cohort = cohort.drop(columns=["_is_positive"])
        return cohort

    # ------------------------------------------------------------------ #
    #  Labels  —  event-based windowed labels via LabelGenerator
    # ------------------------------------------------------------------ #
    def create_labels(self, cohort):
        from label_generator import LabelGenerator

        gen = LabelGenerator(Config.DATA_PATH_ICU, Config.DATA_PATH_HOSP)

        # One prediction point per stay: end of the 24-h lookback window
        prediction_times = pd.DataFrame({
            "stay_id":         cohort["stay_id"].values,
            "prediction_time": cohort["intime"] + timedelta(hours=Config.LOOKBACK_WINDOW_HOURS),
        })

        labels = gen.generateLabels(cohort, prediction_times)
        labels = labels[["stay_id", "label"]].copy()

        pos = labels["label"].sum()
        print(f"Windowed labels — Positive: {int(pos)}/{len(labels)} "
              f"({100*pos/len(labels):.1f}%)")
        return labels

    # ------------------------------------------------------------------ #
    #  Static features
    # ------------------------------------------------------------------ #
    def extract_static_features(self, cohort):
        sf = cohort[["subject_id", "hadm_id", "stay_id"]].copy()

        for col in ["gender", "admission_type", "insurance", "marital_status", "race"]:
            if col in cohort.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    sf[col] = self.label_encoders[col].fit_transform(
                        cohort[col].fillna("Unknown")
                    )
                else:
                    known = set(self.label_encoders[col].classes_)
                    sf[col] = self.label_encoders[col].transform(
                        cohort[col].fillna("Unknown").apply(
                            lambda x: x if x in known else "Unknown"
                        )
                    )

        if "anchor_age" in cohort.columns:
            sf["anchor_age"] = cohort["anchor_age"].fillna(cohort["anchor_age"].median())

        return sf

    def normalize_static_features(self, static_features, fit=True):
        """Z-score normalise continuous static columns (anchor_age)."""
        continuous_cols = [c for c in ["anchor_age"] if c in static_features.columns]
        if not continuous_cols:
            return static_features

        values = static_features[continuous_cols].values.astype(np.float32)

        if fit:
            self.scalers["static"] = StandardScaler()
            values_scaled = self.scalers["static"].fit_transform(values)
        else:
            values_scaled = self.scalers["static"].transform(values)

        sf = static_features.copy()
        for i, col in enumerate(continuous_cols):
            sf[col] = values_scaled[:, i]
        return sf

    # ------------------------------------------------------------------ #
    #  Time-series features
    # ------------------------------------------------------------------ #
    def extract_time_series(self, cohort):
        stay_ids    = cohort["stay_id"].unique().tolist()
        subject_ids = cohort["subject_id"].unique().tolist()

        print(f"Extracting time series for {len(stay_ids)} stays...")

        chartevents = self.data_loader.load_chartevents(
            itemids=Config.VITAL_ITEMIDS, stay_ids=stay_ids
        )
        labevents = self.data_loader.load_labevents(
            itemids=Config.LAB_ITEMIDS, subject_ids=subject_ids
        )

        time_series_data = []
        print(f"Processing {len(cohort)} patient records...")

        for count, (_, row) in enumerate(cohort.iterrows()):
            if (count + 1) % 500 == 0:
                print(f"  {count+1}/{len(cohort)}")

            stay_id    = row["stay_id"]
            intime     = row["intime"]
            window_end = intime + timedelta(hours=Config.LOOKBACK_WINDOW_HOURS)

            all_stay_vitals = chartevents[chartevents["stay_id"] == stay_id].copy()
            vitals = all_stay_vitals[
                (all_stay_vitals["charttime"] >= intime) &
                (all_stay_vitals["charttime"] <  window_end)
            ].copy()

            labs = labevents[
                (labevents["subject_id"] == row["subject_id"]) &
                (labevents["hadm_id"]    == row["hadm_id"])
            ].copy()
            labs = labs[
                (labs["charttime"] >= intime) &
                (labs["charttime"] <  window_end)
            ]

            vitals["hours_from_intime"] = (vitals["charttime"] - intime).dt.total_seconds() / 3600
            labs["hours_from_intime"]   = (labs["charttime"]   - intime).dt.total_seconds() / 3600

            time_bins = np.arange(
                0, Config.LOOKBACK_WINDOW_HOURS + Config.TIME_STEP_HOURS, Config.TIME_STEP_HOURS
            )

            ts_record = {
                "subject_id": row["subject_id"],
                "hadm_id":    row["hadm_id"],
                "stay_id":    stay_id,
                "time_steps": [],
            }

            for t in range(len(time_bins) - 1):
                t_start    = time_bins[t]
                t_end      = time_bins[t + 1]
                vitals_bin = vitals[(vitals["hours_from_intime"] >= t_start) & (vitals["hours_from_intime"] < t_end)]
                labs_bin   = labs[(labs["hours_from_intime"] >= t_start) & (labs["hours_from_intime"] < t_end)]

                features = {}
                for itemid in Config.VITAL_ITEMIDS:
                    item_data = vitals_bin[vitals_bin["itemid"] == itemid]["valuenum"]
                    features[f"vital_{itemid}"] = float(item_data.mean()) if len(item_data) > 0 else np.nan
                for itemid in Config.LAB_ITEMIDS:
                    item_data = labs_bin[labs_bin["itemid"] == itemid]["valuenum"]
                    features[f"lab_{itemid}"] = float(item_data.mean()) if len(item_data) > 0 else np.nan

                features["time_idx"] = t
                ts_record["time_steps"].append(features)

            fv, fm = self._extract_future_vitals(all_stay_vitals, intime, window_end)
            ts_record["future_vitals"] = fv
            ts_record["future_mask"]   = fm
            time_series_data.append(ts_record)

        print("Time series extraction complete.")
        return time_series_data

    def _extract_future_vitals(self, vitals_df, intime, window_end):
        horizon_steps  = Config.TRAJECTORY_HORIZON_STEPS
        step           = timedelta(minutes=Config.PREDICTION_STEP_MINUTES)
        future_vitals  = np.zeros((horizon_steps, Config.NUM_TARGET_VITALS), dtype=np.float32)
        future_mask    = np.zeros((horizon_steps, Config.NUM_TARGET_VITALS), dtype=np.float32)
        target_itemids = list(Config.VITAL_ITEMID_TO_TARGET.keys())
        vital_idx_map  = {iid: i for i, iid in enumerate(target_itemids)}

        horizon_end = window_end + timedelta(hours=Config.PREDICTION_HORIZON_HOURS)
        future_df   = vitals_df[
            (vitals_df["charttime"] >= window_end) &
            (vitals_df["charttime"] <  horizon_end) &
            (vitals_df["itemid"].isin(target_itemids))
        ].copy()

        if future_df.empty:
            return future_vitals, future_mask

        for step_idx in range(horizon_steps):
            bin_start = window_end + step_idx * step
            bin_end   = bin_start + step
            bin_data  = future_df[(future_df["charttime"] >= bin_start) & (future_df["charttime"] < bin_end)]
            for itemid, v_idx in vital_idx_map.items():
                vals = bin_data[bin_data["itemid"] == itemid]["valuenum"].dropna()
                if len(vals) > 0:
                    future_vitals[step_idx, v_idx] = float(vals.mean())
                    future_mask[step_idx, v_idx]   = 1.0

        return future_vitals, future_mask

    # ------------------------------------------------------------------ #
    #  Normalisation
    # ------------------------------------------------------------------ #
    def normalize_features(self, time_series_data, fit=True):
        # 1. Time-series aggregated features
        all_features  = []
        feature_names = None

        for record in time_series_data:
            for ts in record["time_steps"]:
                if feature_names is None:
                    feature_names = [k for k in ts.keys() if k != "time_idx"]
                all_features.append([ts.get(fn, np.nan) for fn in feature_names])

        all_features = np.array(all_features, dtype=np.float32)

        if fit:
            self.scalers["time_series"] = StandardScaler()
            scaled = self.scalers["time_series"].fit_transform(all_features)
        else:
            scaled = self.scalers["time_series"].transform(all_features)

        scaled = np.nan_to_num(scaled, nan=0.0)

        idx = 0
        for record in time_series_data:
            for ts in record["time_steps"]:
                for j, fn in enumerate(feature_names):
                    ts[fn] = scaled[idx, j]
                idx += 1

        # 2. Future vitals dedicated scaler
        all_fv = []
        for record in time_series_data:
            fv   = record.get("future_vitals")
            fmsk = record.get("future_mask")
            if fv is None:
                continue
            for s in range(fv.shape[0]):
                if fmsk[s].any():
                    all_fv.append(fv[s])

        if all_fv:
            all_fv_arr = np.array(all_fv, dtype=np.float32)
            if fit:
                self.scalers["future_vitals"] = StandardScaler()
                self.scalers["future_vitals"].fit(all_fv_arr)

            if "future_vitals" in self.scalers:
                sc = self.scalers["future_vitals"]
                for record in time_series_data:
                    fv   = record.get("future_vitals")
                    fmsk = record.get("future_mask")
                    if fv is None:
                        continue
                    fv_scaled = sc.transform(fv)
                    fv_scaled = np.nan_to_num(fv_scaled, nan=0.0)
                    fv_scaled[fmsk == 0] = 0.0
                    record["future_vitals"] = fv_scaled.astype(np.float32)

        return time_series_data, feature_names

    # ------------------------------------------------------------------ #
    #  Sequence assembly
    # ------------------------------------------------------------------ #
    def create_sequences(self, time_series_data, static_features, labels):
        sequences  = []
        static_idx = static_features.set_index("stay_id")
        labels_idx = labels.set_index("stay_id")

        for record in time_series_data:
            stay_id = record["stay_id"]
            if stay_id not in static_idx.index or stay_id not in labels_idx.index:
                continue

            static_row    = static_idx.loc[stay_id].drop(labels=["subject_id", "hadm_id"], errors="ignore")
            static_values = static_row.values.astype(np.float32)

            label_row   = labels_idx.loc[stay_id]
            label_value = float(
                label_row["label"].iloc[0] if isinstance(label_row, pd.DataFrame) else label_row["label"]
            )

            ts_values = np.array(
                [[ts[k] for k in sorted(ts.keys()) if k != "time_idx"] for ts in record["time_steps"]],
                dtype=np.float32,
            )

            sequences.append({
                "static":        static_values,
                "time_series":   ts_values,
                "label":         label_value,
                "stay_id":       stay_id,
                "future_vitals": record.get("future_vitals"),
                "future_mask":   record.get("future_mask"),
            })

        return sequences

    # ------------------------------------------------------------------ #
    #  Save all fitted artefacts
    # ------------------------------------------------------------------ #
    def save_artefacts(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        for name, key in [("scaler.pkl", "time_series"),
                           ("future_vitals_scaler.pkl", "future_vitals"),
                           ("static_scaler.pkl", "static")]:
            if key in self.scalers:
                joblib.dump(self.scalers[key], os.path.join(save_path, name))
        if self.label_encoders:
            joblib.dump(self.label_encoders, os.path.join(save_path, "label_encoders.pkl"))
        print(f"All artefacts saved to {save_path}")

    # ------------------------------------------------------------------ #
    #  Cache helpers
    # ------------------------------------------------------------------ #
    def save_sequences(self, sequences, path=None):
        path = path or Config.SEQ_CACHE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(sequences, f)
        print(f"Sequences cached to {path}")

    def load_sequences(self, path=None):
        path = path or Config.SEQ_CACHE
        if not os.path.exists(path):
            return None
        print(f"Loading cached sequences from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
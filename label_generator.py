import pandas as pd
import numpy as np
from datetime import timedelta
from config import Config

VASOPRESSOR_ITEMIDS = [
    221906,  # norepinephrine
    221289,  # epinephrine
    221662,  # dopamine
    221749,  # phenylephrine
    222315,  # vasopressin
    221986,  # milrinone
    229617,  # angiotensin ii
]

VENTILATION_ITEMIDS = [
    225792,  # invasive os ventilation
    225794,  # non-invasive ventilation
    224385,  # intubation
    226237,  # reintubation
]

class LabelGenerator:
    def __init__(self, icu_path, hosp_path):
        self.icu_path = icu_path
        self.hosp_path = hosp_path
        self.prediction_horizon = timedelta(hours=Config.PREDICTION_HORIZON_HOURS)

    def _loadVasopressorEvents(self, stay_ids):
        chunks = []
        for chunk in pd.read_csv(
            f"{self.icu_path}inputevents.csv",
            usecols=["stay_id", "starttime", "itemid", "statusdescription"],
            parse_dates=["starttime"],
            chunksize=100_000,
        ):
            chunk = chunk[
                chunk["stay_id"].isin(stay_ids)
                & chunk["itemid"].isin(VASOPRESSOR_ITEMIDS)
                & (chunk["statusdescription"] != "Rewritten")
            ]
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns={"starttime": "event_time"})
        df["event_type"] = "vasopressor"
        return df[["stay_id", "event_time", "event_type"]]

    def _loadVentilationEvents(self, stay_ids):
        chunks = []
        for chunk in pd.read_csv(
            f"{self.icu_path}procedureevents.csv",
            usecols=["stay_id", "starttime", "itemid", "statusdescription"],
            parse_dates=["starttime"],
            chunksize=100_000,
        ):
            chunk = chunk[
                chunk["stay_id"].isin(stay_ids)
                & chunk["itemid"].isin(VENTILATION_ITEMIDS)
                & (chunk["statusdescription"] != "Rewritten")
            ]
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns={"starttime": "event_time"})
        df["event_type"] = "ventilation"
        return df[["stay_id", "event_time", "event_type"]]

    def _loadDeathEvents(self, cohort):
        admissions = pd.read_csv(
            f"{self.hosp_path}admissions.csv",
            usecols=["hadm_id", "deathtime"],
            parse_dates=["deathtime"],
        )
        merged = cohort[["stay_id", "hadm_id"]].merge(admissions, on="hadm_id", how="left")
        merged = merged.dropna(subset=["deathtime"])
        merged = merged.rename(columns={"deathtime": "event_time"})
        merged["event_type"] = "death"
        return merged[["stay_id", "event_time", "event_type"]]

    def generateEventTable(self, cohort):
        stay_ids = set(cohort["stay_id"].tolist())

        vasopressors = self._loadVasopressorEvents(stay_ids)
        ventilation = self._loadVentilationEvents(stay_ids)
        deaths = self._loadDeathEvents(cohort)

        events = pd.concat([vasopressors, ventilation, deaths], ignore_index=True)
        events = events.sort_values(["stay_id", "event_time"]).reset_index(drop=True)
        return events

    def generateLabels(self, cohort, prediction_times):
        """
        For each (stay_id, prediction_time) pair, determine:
          - label: 1 if any deterioration event occurs within prediction horizon
          - time_to_event_hours: hours until first event (NaN if no event)
          - event_type: which event triggered the label

        Args:
            cohort: DataFrame with stay_id, intime, outtime
            prediction_times: DataFrame with stay_id, prediction_time columns

        Returns:
            DataFrame with stay_id, prediction_time, label, time_to_event_hours, event_type
        """
        events = self.generateEventTable(cohort)
        event_lookup = events.groupby("stay_id")

        results = []
        for _, row in prediction_times.iterrows():
            stay_id = row["stay_id"]
            t = row["prediction_time"]
            horizon_end = t + self.prediction_horizon

            if stay_id not in event_lookup.groups:
                results.append({
                    "stay_id": stay_id,
                    "prediction_time": t,
                    "label": 0,
                    "time_to_event_hours": np.nan,
                    "event_type": None,
                })
                continue

            stay_events = event_lookup.get_group(stay_id)
            future_events = stay_events[
                (stay_events["event_time"] > t)
                & (stay_events["event_time"] <= horizon_end)
            ]

            if len(future_events) == 0:
                results.append({
                    "stay_id": stay_id,
                    "prediction_time": t,
                    "label": 0,
                    "time_to_event_hours": np.nan,
                    "event_type": None,
                })
            else:
                first_event = future_events.sort_values("event_time").iloc[0]
                time_to_event = (first_event["event_time"] - t).total_seconds() / 3600
                results.append({
                    "stay_id": stay_id,
                    "prediction_time": t,
                    "label": 1,
                    "time_to_event_hours": round(time_to_event, 4),
                    "event_type": first_event["event_type"],
                })

        return pd.DataFrame(results)

    def generatePredictionTimes(self, cohort, step_minutes=5):
        """
        For each ICU stay, generate a prediction timestamp every step_minutes
        from intime to (outtime - prediction_horizon), so we always have a
        valid future window to label.
        """
        step = timedelta(minutes=step_minutes)
        rows = []
        for _, stay in cohort.iterrows():
            t = stay["intime"] + timedelta(hours=Config.LOOKBACK_WINDOW_HOURS)
            end = stay["outtime"] - self.prediction_horizon
            while t <= end:
                rows.append({"stay_id": stay["stay_id"], "prediction_time": t})
                t += step
        return pd.DataFrame(rows)

    def summarizeLabels(self, labels):
        total = len(labels)
        positive = labels["label"].sum()
        print(f"Total prediction points : {total:,}")
        print(f"Positive (deterioration): {int(positive):,} ({100 * positive / total:.2f}%)")
        print(f"Negative                : {int(total - positive):,}")
        print("\nEvent type breakdown:")
        print(labels[labels["label"] == 1]["event_type"].value_counts())

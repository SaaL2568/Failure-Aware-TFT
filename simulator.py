import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
from config import Config

VITAL_NORMAL_RANGES = {
    "heart_rate":  {"mean": 80,   "std": 12,  "low": 40,  "high": 150},
    "sbp":         {"mean": 120,  "std": 15,  "low": 70,  "high": 200},
    "dbp":         {"mean": 75,   "std": 10,  "low": 40,  "high": 120},
    "resp_rate":   {"mean": 16,   "std": 3,   "low": 6,   "high": 40},
    "spo2":        {"mean": 97,   "std": 1.5, "low": 80,  "high": 100},
    "temperature": {"mean": 37.0, "std": 0.5, "low": 34,  "high": 41},
    "map":         {"mean": 90,   "std": 12,  "low": 50,  "high": 130},
}

DETERIORATION_DRIFT = {
    "heart_rate":  +25,
    "sbp":         -30,
    "dbp":         -20,
    "resp_rate":   +8,
    "spo2":        -6,
    "map":         -25,
}

class SyntheticPatient:
    """
    Generates a realistic ICU vital sign time series for one synthetic patient.
    Optionally introduces a deterioration event at a random time.
    """

    def __init__(self, patient_id, will_deteriorate=False, deterioration_onset_hours=None):
        self.patient_id = patient_id
        self.will_deteriorate = will_deteriorate
        self.deterioration_onset_hours = deterioration_onset_hours or np.random.uniform(4, 12)
        self.admit_time = datetime.now()
        self._state = {v: np.random.normal(r["mean"], r["std"] * 0.3)
                       for v, r in VITAL_NORMAL_RANGES.items()}

    def _clamp(self, value, vital_name):
        r = VITAL_NORMAL_RANGES[vital_name]
        return float(np.clip(value, r["low"], r["high"]))

    def getReading(self, hours_since_admit):
        reading = {"patient_id": self.patient_id, "timestamp": self.admit_time + timedelta(hours=hours_since_admit)}

        deteriorating = self.will_deteriorate and hours_since_admit >= self.deterioration_onset_hours
        drift_progress = 0.0
        if deteriorating:
            drift_progress = min(1.0, (hours_since_admit - self.deterioration_onset_hours) / 2.0)

        for vital, ranges in VITAL_NORMAL_RANGES.items():
            noise = np.random.normal(0, ranges["std"] * 0.15)
            drift = DETERIORATION_DRIFT.get(vital, 0) * drift_progress if deteriorating else 0
            self._state[vital] = self._state[vital] * 0.85 + (ranges["mean"] + drift + noise) * 0.15
            reading[vital] = self._clamp(self._state[vital], vital)

        reading["hours_since_admit"] = hours_since_admit
        reading["true_label"] = int(
            self.will_deteriorate
            and 0 <= (self.deterioration_onset_hours - hours_since_admit) <= Config.PREDICTION_HORIZON_HOURS
        )
        reading["true_time_to_event"] = max(0.0, self.deterioration_onset_hours - hours_since_admit) \
            if self.will_deteriorate else None

        return reading


class ICUStreamSimulator:
    """
    Simulates a live ICU stream by replaying synthetic patients
    at a configurable speed multiplier.

    Usage:
        sim = ICUStreamSimulator(num_patients=10, deterioration_rate=0.3)
        for reading in sim.stream(speed_multiplier=60):
            # reading is a dict ready to POST to the inference API
            print(reading)
    """

    def __init__(self, num_patients=10, deterioration_rate=0.3):
        self.num_patients = num_patients
        self.deterioration_rate = deterioration_rate
        self.patients = self._generatePatients()

    def _generatePatients(self):
        patients = []
        for i in range(self.num_patients):
            will_det = np.random.random() < self.deterioration_rate
            p = SyntheticPatient(
                patient_id=f"SYN_{i:04d}",
                will_deteriorate=will_det,
                deterioration_onset_hours=np.random.uniform(3, 10) if will_det else None,
            )
            patients.append(p)
        print(f"Generated {self.num_patients} patients "
              f"({sum(p.will_deteriorate for p in patients)} will deteriorate)")
        return patients

    def stream(self, duration_hours=24, speed_multiplier=60, step_minutes=5):
        """
        Generator that yields vital sign readings one by one in time order.

        Args:
            duration_hours    : how many simulated hours to run
            speed_multiplier  : 60 = 1 real second per simulated minute
            step_minutes      : how often vitals are emitted per patient
        """
        step_hours = step_minutes / 60
        real_sleep = (step_minutes * 60) / speed_multiplier

        current_hour = 0.0
        while current_hour < duration_hours:
            for patient in self.patients:
                reading = patient.getReading(current_hour)
                yield reading

            time.sleep(real_sleep)
            current_hour += step_hours

    def buildHistoricalWindow(self, patient, current_hours, window_hours=None):
        """
        Build a rolling window of past readings for a patient up to current_hours.
        Used to construct model input from the stream.
        """
        window_hours = window_hours or (Config.OBSERVATION_WINDOW_MINUTES / 60)
        step = Config.PREDICTION_STEP_MINUTES / 60

        readings = []
        t = max(0.0, current_hours - window_hours)
        while t <= current_hours:
            readings.append(patient.getReading(t))
            t += step

        return pd.DataFrame(readings)

    def generateBatchForInference(self, window_hours=None):
        """
        Returns one snapshot of all current patients with their feature windows.
        Useful for batch testing the inference API.
        """
        window_hours = window_hours or (Config.OBSERVATION_WINDOW_MINUTES / 60)
        snapshots = []
        for patient in self.patients:
            onset = patient.deterioration_onset_hours if patient.will_deteriorate else 99
            window_df = self.buildHistoricalWindow(patient, onset - 0.5, window_hours)
            snapshots.append({
                "patient_id": patient.patient_id,
                "will_deteriorate": patient.will_deteriorate,
                "window": window_df,
            })
        return snapshots


def streamToApi(api_url="http://localhost:8000/predict", num_patients=5, duration_hours=6, speed_multiplier=120):
    """
    Convenience function: streams synthetic patient data and POSTs each
    reading to the FastAPI inference endpoint.
    """
    import requests

    sim = ICUStreamSimulator(num_patients=num_patients, deterioration_rate=0.4)

    print(f"Streaming to {api_url}...")
    for reading in sim.stream(duration_hours=duration_hours, speed_multiplier=speed_multiplier):
        payload = {k: v for k, v in reading.items() if k not in ("true_label", "true_time_to_event")}
        try:
            response = requests.post(api_url, json=payload, timeout=2)
            result = response.json()
            if result.get("willDeteriorate"):
                print(f"[ALERT] Patient {reading['patient_id']} | "
                      f"risk={result['riskScore']:.2f} | "
                      f"ETA={result['timeToEventHours']:.1f}h | "
                      f"conf={result['confidence']:.2f}")
        except Exception as e:
            print(f"API error: {e}")


if __name__ == "__main__":
    sim = ICUStreamSimulator(num_patients=5, deterioration_rate=0.4)
    count = 0
    for reading in sim.stream(duration_hours=2, speed_multiplier=300, step_minutes=5):
        print(f"[{reading['timestamp'].strftime('%H:%M')}] "
              f"Patient {reading['patient_id']} | "
              f"HR={reading['heart_rate']:.0f} | "
              f"SBP={reading['sbp']:.0f} | "
              f"SpO2={reading['spo2']:.1f} | "
              f"label={reading['true_label']}")
        count += 1
        if count >= 50:
            break

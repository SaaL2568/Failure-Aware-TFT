from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import numpy as np
import os
from config import Config
from inference import InferencePipeline
from monitoring import DriftDetector

app = FastAPI(
    title="ICU Deterioration Prediction API",
    description="Real-time patient deterioration prediction using Failure-Aware TFT",
    version="1.0.0",
)

pipeline: Optional[InferencePipeline] = None
drift_detector: Optional[DriftDetector] = None

CHECKPOINT_PATH = os.path.join(Config.SAVE_PATH, "best_model.pt")
SCALER_PATH     = os.path.join(Config.SAVE_PATH, "scaler.pkl")
STATIC_SIZE     = len(Config.STATIC_FEATURES)

class VitalReading(BaseModel):
    heart_rate:  Optional[float] = None
    sbp:         Optional[float] = None
    dbp:         Optional[float] = None
    resp_rate:   Optional[float] = None
    spo2:        Optional[float] = None
    temperature: Optional[float] = None
    map:         Optional[float] = None
    creatinine:  Optional[float] = None
    lactate:     Optional[float] = None
    timestamp:   Optional[str]   = None

class PatientRequest(BaseModel):
    patient_id:        str
    static_features:   Dict[str, Any] = Field(default_factory=dict)
    vital_window:      list[VitalReading]

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P001",
                "static_features": {
                    "gender": 1,
                    "anchor_age": 65,
                    "admission_type": 0,
                    "insurance": 1,
                    "marital_status": 0,
                    "race": 0,
                },
                "vital_window": [
                    {"heart_rate": 88, "sbp": 112, "dbp": 72, "resp_rate": 18, "spo2": 96},
                    {"heart_rate": 92, "sbp": 108, "dbp": 70, "resp_rate": 20, "spo2": 95},
                ],
            }
        }

class PredictionResponse(BaseModel):
    patient_id:       str
    willDeteriorate:  bool
    timeToEventHours: float
    confidence:       float
    riskScore:        float
    driftAlert:       bool = False
    timestamp:        str

@app.on_event("startup")
async def loadModel():
    global pipeline, drift_detector

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}. Predictions will fail until model is trained.")
        return

    ts_feature_cols = (
        len([
            f"{v}_{s}"
            for v in ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature", "map"]
            for s in ["mean", "min", "max", "std", "slope", "last", "delta", "missing", "count", "above_high", "below_low"]
        ])
        + len([
            f"{l}_{s}"
            for l in ["creatinine", "lactate", "potassium", "sodium"]
            for s in ["mean", "min", "max", "std", "slope", "last", "delta", "missing", "count", "above_high", "below_low"]
        ])
        + 1  # time_since_admission_hours
    )

    pipeline = InferencePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        scaler_path=SCALER_PATH,
        static_size=STATIC_SIZE,
        time_series_size=ts_feature_cols,
    )
    drift_detector = DriftDetector(window_size=Config.DRIFT_WINDOW_SIZE)
    print("Model loaded and API ready.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PatientRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    vitals_dicts = [r.dict() for r in request.vital_window]
    if len(vitals_dicts) == 0:
        raise HTTPException(status_code=422, detail="vital_window must contain at least one reading.")

    import pandas as pd
    window_df = pd.DataFrame(vitals_dicts).drop(columns=["timestamp"], errors="ignore")
    window_df = window_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    result = pipeline.predictFromRawWindow(window_df, request.static_features)

    drift_alert = False
    if drift_detector is not None:
        drift_detector.update(result["riskScore"])
        drift_alert = drift_detector.isDrifting()

    from datetime import datetime
    return PredictionResponse(
        patient_id=request.patient_id,
        willDeteriorate=result["willDeteriorate"],
        timeToEventHours=result["timeToEventHours"],
        confidence=result["confidence"],
        riskScore=result["riskScore"],
        driftAlert=drift_alert,
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "device": str(Config.DEVICE),
    }

@app.get("/")
async def root():
    return {"message": "ICU Deterioration Prediction API — POST to /predict"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=Config.API_HOST, port=Config.API_PORT, reload=False)

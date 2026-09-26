"""
api/main.py

FastAPI skeleton: one POST endpoint /detect taking a channel ID (uses stored
test data) or a raw array (new telemetry), returning decision + score.
Also exposes /rank for the multi-channel "which channel is responsible" view.

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install fastapi uvicorn
    python models/train.py                      # train weights first (see below)
    uvicorn api.main:app --reload --port 8000
Then open dashboard/index.html in a browser (it points at
http://localhost:8000 by default) or test with curl:

    curl -X POST http://localhost:8000/detect \
         -H "Content-Type: application/json" \
         -d "{\"channel_id\": \"P-1\"}"

    curl -X POST http://localhost:8000/rank \ 
         -H "Content-Type: application/json" \
         -d "{\"channel_ids\": [\"P-1\", \"S-1\", \"E-1\"]}"
=============================================================================
"""

import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline import run_pipeline, run_pipeline_multi

app = FastAPI(title="SATSENTINEL API", version="0.1.0")


class DetectRequest(BaseModel):
    channel_id: str
    raw_array: Optional[List[float]] = None


class RankRequest(BaseModel):
    channel_ids: List[str]


@app.get("/")
def health():
    """Simple liveness check -- hit this first to confirm the server is up."""
    return {"status": "ok", "service": "SATSENTINEL"}


@app.post("/detect")
def detect(req: DetectRequest):
    """Runs the full AE+IF pipeline for one channel and returns the
    per-window decisions, anomaly scores, and the raw reconstruction-error
    series (what the dashboard plots).
    """
    try:
        result = run_pipeline(req.channel_id, raw_array=req.raw_array)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "channel": result["channel"],
        "decision": result["decision"],
        "scores": result["scores"],
        "max_score": result["max_score"],
        "mean_score": result["mean_score"],
        "error_series": result["test_error_series"],
    }


@app.post("/rank")
def rank(req: RankRequest):
    """Runs /detect-equivalent logic across several channels at once and
    sorts them by anomaly severity, for the 'which channel is responsible'
    multi-channel view.
    """
    try:
        result = run_pipeline_multi(req.channel_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "ranking": [
            {"channel": cid, "max_score": max_s, "mean_score": mean_s}
            for cid, max_s, mean_s in result["ranking"]
        ]
    }

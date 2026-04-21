# !usr/bin/env/python3

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
SAVED_DIR = BASE_DIR / "saved"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import Inference 
from helpers import load_json

app = Flask(__name__)



def _compute_model_metrics(metrics_payload: dict[str, Any]) -> dict[str, float]:
    test_metrics = metrics_payload.get("test") or {}
    tn = float(test_metrics.get("tn", 0.0))
    fp = float(test_metrics.get("fp", 0.0))
    tp = float(test_metrics.get("tp", 0.0))
    fn = float(test_metrics.get("fn", 0.0))
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else float(test_metrics.get("recall", 0.0))
    youden_j = sensitivity + specificity - 1.0
    return {
        "f1": float(test_metrics.get("f1", 0.0)),
        "youden_j": round(youden_j, 5),
        "auc_roc": float(test_metrics.get("roc_auc", 0.0)),
    }

@lru_cache(maxsize=1)
def get_metrics() -> dict[str, float]:
    return _compute_model_metrics(load_json(SAVED_DIR / "model" / "metrics.json"))



@lru_cache(maxsize=1)
def get_service() -> Inference:
    return Inference(project_root=BASE_DIR)

def predict(text1: str, text2: str) -> dict[str, Any]:
    return get_service().predict(text1, text2).to_dict()


@app.route("/", methods=["GET"])
def home_route():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    text1 = (data.get("text1") or "").strip()
    text2 = (data.get("text2") or "").strip()
    if not text1 or not text2: return jsonify({"error": "Both text fields are required."}), 400
    try: result = predict(text1, text2)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    return jsonify(result)

@app.route("/metrics", methods=["GET"])
def metrics_route():
    try:
        return jsonify(get_metrics())
    except Exception as exc:
        return jsonify({"error": f"Failed to load metrics: {exc}"}), 500


# ping for cron job
@app.route("/ping")
def ping():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)

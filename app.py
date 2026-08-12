"""
Battery Capacity Prediction — Flask Web Application
Predicts lithium-ion battery capacity using a trained regression model.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "finalmodel.sav"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ---------------------------------------------------------------------------
# Load Model at Startup
# ---------------------------------------------------------------------------
if MODEL_PATH.exists():
    loaded_model = pickle.load(open(MODEL_PATH, "rb"))
    MODEL_LOADED = True
else:
    MODEL_LOADED = False
    print(f"[WARNING] Model file not found at {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def main():
    """Landing page with prediction form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    """Handle form submission and return prediction."""
    if not MODEL_LOADED:
        return "Model not loaded. Please ensure finalmodel.sav is present.", 503

    cycle = float(request.form["cycle"])
    voltage_battery = float(request.form["voltage_battery"])
    temp_battery = float(request.form["temp_battery"])
    time = float(request.form["time"])

    input_data = {
        "cycle": [cycle],
        "voltage_battery": [voltage_battery],
        "temp_battery": [temp_battery],
        "time": [time],
    }

    Xnew = pd.DataFrame(input_data)
    ynew = loaded_model.predict(Xnew)

    return render_template("result.html", predicted_value=ynew)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

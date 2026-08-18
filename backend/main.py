from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CKPT = "checkpoint"

# Load tabular models
model_stageA = joblib.load(f"{CKPT}/stageA_failure_predictor.pkl")
model_stageB = joblib.load(f"{CKPT}/stageB_failure_type.pkl")
iso_forest = joblib.load(f"{CKPT}/isolation_forest_anomaly.pkl")
feature_cols = joblib.load(f"{CKPT}/feature_cols.pkl")
le_failtype = joblib.load(f"{CKPT}/failure_type_encoder.pkl")

fleet_df = pd.read_csv(f"{CKPT}/fleet_1000_predictions.csv")

# --- CV / image endpoint disabled for now — TensorFlow not installed on this Python version ---
# import tensorflow as tf
# cv_model = tf.keras.models.load_model(f"{CKPT}/fastener_classifier_final.keras")
# class_names = ["Defective", "Non Defective"]

@app.get("/trains")
def list_trains():
    return fleet_df[["train_id", "priority_bucket", "failure_probability", "predicted_failure_type"]].to_dict(orient="records")

@app.get("/trains/{train_id}")
def get_train(train_id: int):
    row = fleet_df[fleet_df["train_id"] == train_id]
    if row.empty:
        return {"error": "Train not found"}
    return row.to_dict(orient="records")[0]

# --- Uncomment once TensorFlow is available in this environment ---
# @app.post("/predict/image")
# async def predict_image(file: UploadFile = File(...)):
#     ...
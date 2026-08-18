from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cv_model = tf.keras.models.load_model("checkpoint/fastener_classifier_final.keras")
class_names = ["Defective", "Non Defective"]  # confirm this order matches your training run's class_names output

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img), axis=0)

    preds = cv_model.predict(arr)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return {
        "predicted_class": class_names[class_idx],
        "confidence": confidence
    }
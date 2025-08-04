from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image

# Load model
MODEL_PATH = "breast_cancer_model.keras"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class labels (change if you have different labels)
CLASS_NAMES = ["Benign", "Malignant"]

# Create FastAPI app
app = FastAPI(title="Breast Cancer Classification API")

@app.get("/")
def home():
    return {"message": "Welcome to Breast Cancer Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # Adjust if your model uses different size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        score = float(predictions[0][0])  # Adjust if softmax for multiple classes

        label = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(score if label == CLASS_NAMES[1] else 1 - score, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

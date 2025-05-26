from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Helper function to preprocess image
def preprocess_image(image: Image.Image) -> np.ndarray:
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS).convert("RGB")
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        data = preprocess_image(image)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])
        
        return JSONResponse(content={
            "class": class_name,
            "confidence_score": confidence_score
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional: Run the app (use only when not deploying via external server)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
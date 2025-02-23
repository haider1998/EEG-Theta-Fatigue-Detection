from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import uvicorn
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="EEG Theta Band Fatigue Detection API", version="1.0")

# Load the pre-trained model
model = load('models/fatigue_detector.pkl')


# Define the input data schema
class EEGData(BaseModel):
    features: list[float]  # List of extracted features


@app.post("/predict/")
async def predict(data: EEGData):
    # Convert input data to numpy array
    input_features = np.array(data.features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)

    # Return the prediction result
    return {"prediction": int(prediction[0])}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)

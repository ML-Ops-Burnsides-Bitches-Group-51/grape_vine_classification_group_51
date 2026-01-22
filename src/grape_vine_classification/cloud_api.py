import os
import torch
from google.cloud import storage
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from torchvision import transforms




BUCKET_NAME = "models_grape_gang"
MODEL_FILE = "cloud_model.pth"
LOCAL_MODEL_PATH = "/tmp/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["Ak","Ala_Idris","Buzgule","Dimnit","Nazli"]

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

class PredictionOutput(BaseModel):
    species: str


def download_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)

    blob.download_to_filename(LOCAL_MODEL_PATH)

    print("Model downloaded complected succefully: ", os.path.isfile(LOCAL_MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    download_model()

    model = torch.load(LOCAL_MODEL_PATH,map_location=device)

    model.to(device)
    model.eval()

    print("model load succefull")
    yield
    del model

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionOutput)
async def predict_species(file: UploadFile = File()):
    
    try:

        image_data = await file.read()
        input_tensor = transform(image_data)
        
        # Add batch dimension if necessary
        if input_tensor.ndimension() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1)
            species = class_names[prediction]

        return PredictionOutput(species=species)

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

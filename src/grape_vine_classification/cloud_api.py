import os
import torch
from google.cloud import storage
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from contextlib import asynccontextmanager
from torchvision import transforms
from PIL import Image
import io
import datetime
import json

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

def save_prediction_to_gcp(img: torch.Tensor, outputs: list[float], species: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    time = datetime.datetime.now(tz=datetime.UTC)

    # extract features from transformed input image
    avg_brightness = torch.mean(img).item()    
    contrast = torch.std(img).item()
    gradients = torch.gradient(img, dim=[2,3])
    sharpness = torch.mean(torch.abs(gradients[0]) + torch.abs(gradients[1])).item()

    # construct data to be saved
    data = {  
        "avg_brightness": avg_brightness,  
        "contrast": contrast,   
        "sharpness": sharpness,  
        "timestamp": time.isoformat(), 
        "class_probabilities": outputs,
        "species": species,
    }

    # save data to cloud
    blob = bucket.blob(f"predictions/prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Finished saving image features and model outputs")

def download_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)

    blob.download_to_filename(LOCAL_MODEL_PATH)

    print("Model downloaded complete and available at: ", os.path.isfile(LOCAL_MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    download_model()
    model = torch.load(LOCAL_MODEL_PATH, map_location = device)
    model.eval()
    print("model succefully loaded")

    yield

    del model

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model = PredictionOutput)
async def predict_species(background_tasks: BackgroundTasks, file: UploadFile = File()):
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image)
        
        # Add batch dimension if necessary
        if input_tensor.ndimension() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(outputs, dim=1)
            species = class_names[prediction]

        background_tasks.add_task(save_prediction_to_gcp, input_tensor, outputs.softmax(-1).squeeze().tolist(), species)

        return PredictionOutput(species = species)

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

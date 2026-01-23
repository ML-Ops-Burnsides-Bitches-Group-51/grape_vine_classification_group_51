import json
import os
from pathlib import Path

import pandas as pd
# from evidently.legacy.metric_preset import TargetDriftPreset, TextEvals
# from evidently.legacy.report import Report
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.cloud import storage

MODEL_BUCKET_NAME = "models_grape_gang"
DATA_BUCKET_NAME = "grapevine_data"
REFERENCE_DATA_NAME = "data/processed_dataset/feature_database.csv"
LOCAL_REFERENCE_DATA_PATH = "tmp/feataure_database.csv"

def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()]) # select relevant columns from both
    report_eval = report.run(reference_data=reference_data, current_data=current_data)
    report_eval.save_html("tmp/report.html")

def download_reference_data() -> None:
    client = storage.Client()
    bucket = client.bucket(DATA_BUCKET_NAME)
    blob = bucket.blob(REFERENCE_DATA_NAME)
    blob.download_to_filename(LOCAL_REFERENCE_DATA_PATH)

def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global reference_data
    download_reference_data()
    reference_data = pd.read_csv(LOCAL_REFERENCE_DATA_PATH)
    reference_data = reference_data.rename(columns={'target': 'prediction'})

    yield

    del reference_data


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("tmp/predictions/prediction_*.json")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    features = ["brightness", "contrast", "sharpness", "prediction"]
    rows = []
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            row = {feature: data[feature] for feature in features}
            rows.append(row)
    dataframe = pd.DataFrame(rows)
    return dataframe


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client().bucket(MODEL_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="predictions/prediction_"))
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    os.makedirs("tmp/predictions/", exist_ok=True)
    for blob in latest_blobs:
        blob.download_to_filename("tmp/" + blob.name)


@app.get("/report", response_class = JSONResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    current_data = load_latest_files(Path("."), n=n)
    run_analysis(reference_data, current_data)

    return JSONResponse(
        content={"status": "success", "message": "Report available at tmp/report.html"},
        status_code=200)
from pathlib import Path
from locust import HttpUser, task, between

IMAGE_PATH = Path(__file__).parent / "test_assets" / "processed_sample.png"
assert IMAGE_PATH.exists(), f"Missing test image: {IMAGE_PATH}"


class APIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        with IMAGE_PATH.open("rb") as f:
            files = [("files", ("processed_sample.png", f, "image/png"))]
            self.client.post("/predict?num_predictions=3", files=files)


# To run this locust test, use the command:
# uv run locust -f tests/locustfile.py -H http://127.0.0.1:8000

# Then open your browser and go to:
# http://localhost:8089

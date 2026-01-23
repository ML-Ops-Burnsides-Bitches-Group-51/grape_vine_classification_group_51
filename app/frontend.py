import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    # 1. Try to get from Environment Variable first (works for local and deployed)
    name = os.environ.get("BACKEND", None)
    if name:
        return name

    # 2. Try Google Cloud lookup
    try:
        parent = "projects/grapevine-gang/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        
        for service in services:
            if service.name.split("/")[-1] == "backend":
                return service.uri
    except Exception as e:
        
        # If not logged into gcloud locally, this will fail
        st.warning(f"Cloud lookup failed: {e}. Falling back to localhost.")

    # 3. Final fallback for local development
    return "http://localhost:8000"


def classify_image(image_bytes, backend):
    """Send the image to the backend for classification."""
    
    predict_url = f"{backend}/predict"
    files = [('files', ('image.jpg', image_bytes, 'image/jpeg'))]
    response = requests.post(predict_url, files=files, timeout=10)

    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)
        
        print(f"DEBUG: Backend response was: {result}") 
        if result is not None:
            first_item = result['results'][0]
            prediction = first_item['predicted_label']
            top_k = first_item['top_k']
            probabilities = first_item['probabilities']

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            data = {"Class": [f"{top_k[i]['label']}" for i in range(5)], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
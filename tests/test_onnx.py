import onnxruntime as rt
import numpy as np
import torch
import pytest
import os

from src.grape_vine_classification.model_lightning import SimpleCNN
from tests import model_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Define paths
onnx_filepath = "models/simple_cnn.onnx"
pt_filepath = "models/simple_cnn.pth"


@pytest.mark.skipif(
    (not os.path.exists(onnx_filepath)) or (not os.path.exists(pt_filepath)), reason="One of the models is not found"
)
@pytest.mark.parametrize("onnx_model_file,pytorch_model_file", [(onnx_filepath, pt_filepath)])
def test_onnx_model(
    onnx_model_file: str,
    pytorch_model_file: str,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    random_input = torch.randn(1, 1, 128, 128)
    # Run onnx model
    ort_session = rt.InferenceSession(onnx_model_file)
    ort_inputs = {ort_session.get_inputs()[0].name: random_input.numpy().astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Load config file

    # Run pytorch model
    pytorch_model = SimpleCNN(model_config)
    pytorch_model.load_state_dict(torch.load(pytorch_model_file, map_location=DEVICE))
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outs = pytorch_model(random_input).cpu().numpy()
    # Assert that output is the same within tolerance
    assert np.allclose(ort_outs[0], pytorch_outs, rtol=rtol, atol=atol), "Pytorch and ONNX do not match."

    print("Pytorch and ONNX match.")

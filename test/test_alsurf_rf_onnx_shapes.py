import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "alsurf_rf"
SHAPE_CASES = {
    "dynamic_batch": {
        "inputs": [("chemical_input_raw", ["batch_size", 9])],
        "outputs": [("sorbed_output_raw", ["batch_size", 2])],
    },
    "batch1": {
        "inputs": [("chemical_input_raw", [1, 9])],
        "outputs": [("sorbed_output_raw", [1, 2])],
    },
    "feature_vector": {
        "inputs": [("chemical_input_raw", [9])],
        "outputs": [("sorbed_output_raw", [2])],
    },
    "scalar": {
        "inputs": [
            ("Zn(OH)2(aq)", []),
            ("Zn(OH)3-", []),
            ("Zn(OH)4--", []),
            ("ZnOH+", []),
            ("Zn++", []),
            ("Na+", []),
            ("NO3-", []),
            ("Fe++", []),
            ("O2(aq)", []),
        ],
        "outputs": [("SURF-H+", []), ("SURF-Zn++", [])],
    },
}


@pytest.fixture(scope="module")
def parity_data():
    model = joblib.load(MODEL_DIRECTORY / "alsurf_9.joblib")
    x_scaler = joblib.load(MODEL_DIRECTORY / "x_scalar.joblib")
    y_scaler = joblib.load(MODEL_DIRECTORY / "y_scalar.joblib")
    with (MODEL_DIRECTORY / "metadata.json").open(encoding="utf-8") as stream:
        selected_features = json.load(stream)["selected_features"][8]

    data = pd.read_csv(
        REPOSITORY_ROOT / "data" / "phreeqc_example8_all_simulations.csv"
    ).dropna()
    data = data.rename(
        columns={"specified_pH": "input_pH", "input_Zn_mol_L": "zn"}
    )
    data["h"] = 10.0 ** (-data["input_pH"])

    all_features = list(x_scaler.feature_names_in_)
    selected_indices = [
        all_features.index(feature_name) for feature_name in selected_features
    ]
    raw_inputs = data[all_features].to_numpy(dtype=np.float64)[
        :, selected_indices
    ]
    scaled_inputs = x_scaler.transform(data[all_features])[:, selected_indices]
    scaled_frame = pd.DataFrame(scaled_inputs, columns=selected_features)
    expected_outputs = y_scaler.inverse_transform(model.predict(scaled_frame))
    return selected_features, raw_inputs, expected_outputs


@pytest.mark.parametrize("shape_mode", SHAPE_CASES)
def test_alsurf_9_graph_signature(shape_mode):
    model_path = MODEL_DIRECTORY / f"alsurf_9_{shape_mode}.onnx"
    onnx.checker.check_model(onnx.load(model_path), full_check=True)
    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    actual_inputs = [
        (tensor.name, tensor.shape) for tensor in session.get_inputs()
    ]
    actual_outputs = [
        (tensor.name, tensor.shape) for tensor in session.get_outputs()
    ]
    assert actual_inputs == SHAPE_CASES[shape_mode]["inputs"]
    assert actual_outputs == SHAPE_CASES[shape_mode]["outputs"]
    assert all(tensor.type == "tensor(double)" for tensor in session.get_inputs())
    assert all(tensor.type == "tensor(double)" for tensor in session.get_outputs())


@pytest.mark.parametrize("shape_mode", SHAPE_CASES)
def test_alsurf_9_native_onnx_parity(shape_mode, parity_data):
    selected_features, raw_inputs, expected_outputs = parity_data
    session = ort.InferenceSession(
        MODEL_DIRECTORY / f"alsurf_9_{shape_mode}.onnx",
        providers=["CPUExecutionProvider"],
    )

    if shape_mode == "dynamic_batch":
        actual_outputs = session.run(
            None, {"chemical_input_raw": raw_inputs}
        )[0]
        expected = expected_outputs
    elif shape_mode == "batch1":
        actual_outputs = session.run(
            None, {"chemical_input_raw": raw_inputs[0:1]}
        )[0]
        expected = expected_outputs[0:1]
    elif shape_mode == "feature_vector":
        actual_outputs = session.run(
            None, {"chemical_input_raw": raw_inputs[0]}
        )[0]
        expected = expected_outputs[0]
    else:
        scalar_inputs = {
            feature_name: np.asarray(value, dtype=np.float64)
            for feature_name, value in zip(selected_features, raw_inputs[0])
        }
        actual_outputs = np.asarray(
            [np.asarray(value).item() for value in session.run(None, scalar_inputs)]
        )
        expected = expected_outputs[0]

    np.testing.assert_allclose(
        actual_outputs, expected, rtol=1.0e-12, atol=1.0e-12
    )

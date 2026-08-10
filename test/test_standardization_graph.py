import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from nn_regressor.integrated_scale_nn import IntegratedChemicalRegressor
from nn_regressor.multi_layer_nn import ChemicalRegressor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_nn_scale_graph():
    model_directory = REPOSITORY_ROOT / "models" / "alsurf_nn"
    model = ChemicalRegressor([2, 512, 128, 64, 2])
    model.load_state_dict(
        torch.load(model_directory / "zn_h_regressor_base.pt")
    )
    model.eval()

    x_scaler = joblib.load(model_directory / "x_scalar.joblib")
    y_scaler = joblib.load(model_directory / "y_scalar.joblib")
    integrated_model = IntegratedChemicalRegressor(model, x_scaler, y_scaler)
    integrated_model.eval()

    x_raw = np.array([[1.0e-7, 1.0e-4]], dtype=np.float64)
    
    # Prevent the warning from x_scalar, we utilize the pd.Dataframe 
    # to initialize it. x_scalar expects 'names'
    x_with_names = pd.DataFrame(x_raw, columns=x_scaler.feature_names_in_)
    x_scaled = torch.tensor(
        x_scaler.transform(x_with_names), dtype=torch.float64
    )

    with torch.no_grad():
        y_scaled = model(x_scaled).numpy()
        y_integrated = integrated_model(torch.from_numpy(x_raw)).numpy()

    y_expected = y_scaler.inverse_transform(y_scaled)
    np.testing.assert_allclose(
        y_integrated, y_expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_rf_onnx_graph_is_valid_float64():
    model_path = REPOSITORY_ROOT / "models" / "alsurf_rf" / "alsurf_6.onnx"
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].type == "tensor(double)"
    assert session.get_outputs()[0].type == "tensor(double)"


def test_rf_scale_graph():
    model_directory = REPOSITORY_ROOT / "models" / "alsurf_rf"
    base_model = joblib.load(model_directory / "alsurf_6.joblib")
    x_scaler = joblib.load(model_directory / "x_scalar.joblib")
    y_scaler = joblib.load(model_directory / "y_scalar.joblib")

    with (model_directory / "metadata.json").open(encoding="utf-8") as stream:
        selected_features = json.load(stream)["selected_features"][5]

    data = pd.read_csv(
        REPOSITORY_ROOT / "data" / "phreeqc_example8_all_simulations.csv"
    ).dropna()
    data = data.rename(
        columns={"specified_pH": "input_pH", "input_Zn_mol_L": "zn"}
    )
    data["h"] = 10.0 ** (-data["input_pH"])

    all_features = list(x_scaler.feature_names_in_)
    raw_features = data[all_features]
    scaled_features = pd.DataFrame(
        x_scaler.transform(raw_features),
        columns=all_features,
        index=raw_features.index,
    )
    base_scaled = base_model.predict(scaled_features[selected_features])
    base_raw = y_scaler.inverse_transform(base_scaled)

    session = ort.InferenceSession(
        model_directory / "alsurf_6.onnx",
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_raw = session.run(
        None,
        {input_name: raw_features[selected_features].to_numpy(dtype=np.float64)},
    )[0]

    print(f'RF (Base): {base_raw}')
    print(f'RF (ONNX): {onnx_raw}')
    np.testing.assert_allclose(
        onnx_raw, base_raw, rtol=1.0e-12, atol=1.0e-12
    )

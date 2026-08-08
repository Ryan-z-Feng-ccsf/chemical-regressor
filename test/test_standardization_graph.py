import numpy as np
import torch
import joblib
from nn_regressor.multi_layer_nn import ChemicalRegressor
from nn_regressor.integrated_scale_nn import IntegratedChemicalRegressor

model_path = 'models/zn_h_regressor/zn_h_regressor_base.pt'
x_scalar_path = 'models/zn_h_regressor/x_scalar.joblib'
y_scalar_path = 'models/zn_h_regressor/y_scalar.joblib'

model = ChemicalRegressor([2, 512, 128, 64, 2])
model.load_state_dict(torch.load(model_path))
model.eval()


x_scalar = joblib.load(x_scalar_path)
y_scalar = joblib.load(y_scalar_path)

integrated_model = IntegratedChemicalRegressor(model, x_scalar, y_scalar)
integrated_model.eval()

x_raw = np.array([[1.0e-7,1.0e-4]])  # H+ = 1.0e-7 mol/L, Zn = 1e-4 mol/L

x_scaled = x_scalar.transform(x_raw)

# Get the output from the base model
x_scaled = torch.tensor(x_scaled, dtype=torch.float64)

with torch.no_grad():
    y_scaled = model(x_scaled)

y_scaled = y_scaled.numpy()
y_raw = y_scalar.inverse_transform(y_scaled)
print(f"Base model output: {y_raw}")



# Get the output from the integrated model
x_raw_tensor = torch.tensor(x_raw, dtype=torch.float64)
with torch.no_grad():
    y_raw_integrate = integrated_model(x_raw_tensor)
    
y_raw_integrate = y_raw_integrate.numpy()
print(f"integrated model: {y_raw_integrate}")


try:
    np.testing.assert_allclose(y_raw, y_raw_integrate, rtol=1e-12, atol=1e-12)
except AssertionError as e:
    print("❌ Test fails!")
    print(e)
    

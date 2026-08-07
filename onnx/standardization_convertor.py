import joblib
import torch
from nn_regressor.integrated_scale_nn import IntegratedChemicalRegressor
from nn_regressor.multi_layer_nn import ChemicalRegressor

torch.set_default_dtype(torch.float64)  # Set the global precision as Double
    
model_path = 'models/zn_h_regressor_base.pt'
onnx_file_path = 'models/zn_h_regressor_integrated_batch1.onnx'
x_scalar_path = 'models/zn_h_regressor_scale/x_scalar.joblib'
y_scalar_path = 'models/zn_h_regressor_scale/y_scalar.joblib'


dummy_input = torch.randn(1, 2, dtype=torch.float64)   # We can set it as batch size

model = ChemicalRegressor([2, 512, 128, 64, 2])
model.load_state_dict(torch.load(model_path))
model.eval()    # Shut down dropout

x_scalar = joblib.load(x_scalar_path)
y_scalar = joblib.load(y_scalar_path)

integrated_model = IntegratedChemicalRegressor(model, x_scalar, y_scalar)
integrated_model.eval()


torch.onnx.export(
    integrated_model,
    dummy_input,    # Run through the model to set the weight
    onnx_file_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['chemical_input_raw'],   
    output_names=['sorbed_output_raw'],
    # dynamic_axes={
    #     'chemical_input_raw': {0: 'batch_size'},
    #     'sorbed_output_raw': {0: 'batch_size'}
    # },
    dynamo=False  # ⭐️ Eliminate API runtime error
)
print(f"✅")


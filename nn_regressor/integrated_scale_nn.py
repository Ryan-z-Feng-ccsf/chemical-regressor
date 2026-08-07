import torch.nn as nn
import torch
from typing import Any
class IntegratedChemicalRegressor(nn.Module):
    def __init__(self, model, x_scalar, y_scalar, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = model
        
        self.register_buffer('x_mean',torch.tensor(x_scalar.mean_,dtype=torch.float64))
        self.register_buffer('x_scale', torch.tensor(x_scalar.scale_,dtype=torch.float64))
        
        self.register_buffer('y_mean', torch.tensor(y_scalar.mean_,dtype=torch.float64))
        self.register_buffer('y_scale', torch.tensor(y_scalar.scale_, dtype=torch.float64))
        
    def forward(self, x_raw):
        x_scale = (x_raw - self.x_mean) / self.x_scale
        
        y_scale = self.model(x_scale)
        
        y_raw = y_scale * self.y_scale + self.y_mean
        return y_raw
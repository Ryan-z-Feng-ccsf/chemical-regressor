import torch.nn as nn
import torch
class ChemicalRegressor(nn.Module):
    def __init__(self, layer, p=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(layer) - 1):
            self.layers.append(nn.Linear(layer[i], layer[i + 1],dtype=torch.float64))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x) # Prevent overfitting (Shut down 10% neurons)
            

        x = self.layers[-1](x)

        return x
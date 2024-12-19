import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, model_dimension):
        self.model_dimension = model_dimension

    def forward(self, input_layer):
        return nn.LayerNorm(self.model_dimension, input_layer)
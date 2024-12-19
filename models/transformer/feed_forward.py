import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, model_dimension, hidden_layer_dimension):
        super().__init__()
        self.lin_trans_1 = nn.Linear(model_dimension, hidden_layer_dimension)
        self.lin_trans_2 = nn.Linear(hidden_layer_dimension, model_dimension)
    
    def forward(self, attention_output): #FFN(x) = max(0, xW1 + b1)W2 + b2
        ffo = self.lin_trans_1(attention_output)
        ffo = nn.functional.relu(ffo)
        ffo = self.lin_trans_2(ffo)
        return ffo

        
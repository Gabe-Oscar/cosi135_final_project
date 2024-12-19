import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    #encode positionof embeddings
        

    def __init__(self, model_dimension, max_seq_len):
        super().__init__()
        self.model_dimension = model_dimension
        self.scalar = 10000
        self.pe = self.create_positional_encodings(max_seq_len)
        
        
    def create_positional_encodings(self, seq_len):
        pe = torch.zeros(seq_len, self.model_dimension)    
        for pos in range(seq_len):
            for i in range(self.model_dimension//2):
                denom = math.pow(self.scalar,(2*i/self.model_dimension))
                pe[pos, 2*i] = math.sin(pos/denom)
                pe[pos, 2*i+1] = math.cos(pos/denom)
        return pe
        
    def forward(self, batch):
        return batch + self.pe
    



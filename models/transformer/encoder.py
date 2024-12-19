from torch import nn as nn
from models.transformer import encoder_layer
from models.transformer import positional_encoding
class Encoder(nn.Module):
    def __init__(self, model_dimension, num_layers, hidden_layer_dimension, key_dimension, value_dimension, num_heads, vocab_size, max_seq_len):
        super().__init__()
        self.model_dimension = model_dimension
        self.num_layers = num_layers
        self.hidden_layer_dimension = hidden_layer_dimension
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.layers = nn.ModuleList([encoder_layer.EncoderLayer(model_dimension=model_dimension, hidden_layer_dimension=hidden_layer_dimension, key_dimension=key_dimension, value_dimension=value_dimension, num_heads = num_heads) for _ in range(0, num_layers)])
        self.embedding = nn.Embedding(vocab_size, model_dimension)
        self.positionalEncoding = positional_encoding.PositionalEncoding(self.model_dimension, max_seq_len)



    def forward(self, tokens, masks,sequence_length):
        output = self.embedding(tokens)
        output = self.positionalEncoding(output)
        for layer in self.layers:
            output = layer(output, masks)
        return output

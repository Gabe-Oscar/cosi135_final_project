import torch.nn as nn

from models.transformer.feed_forward import FeedForward
from models.transformer.multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, hidden_layer_dimension, key_dimension, value_dimension, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention_layer = MultiHeadAttention(model_dimension, key_dimension, value_dimension,num_heads)

        self.feed_forward_layer = FeedForward(model_dimension, hidden_layer_dimension)
        self.layer_norm_1 = nn.LayerNorm(model_dimension)
        self.layer_norm_2 = nn.LayerNorm(model_dimension)

    
    def forward(self, x, masks):    
        x2 = self.attention_layer(x,x,x, masks)
        output = self.layer_norm_1(x +  x2)
        x2 = self.feed_forward_layer(output)
        output = self.layer_norm_2(output+x2)
        return output


import torch.nn as nn
from models.transformer.encoder import Encoder
from models.shared_architecture.classification_head import ClassificationHead

class TransformerModel(nn.Module):
    def __init__(self, model_dimension, encoder_layers, hidden_layer_dimension, key_dimension, value_dimension, vocab_size, num_heads, num_labels, max_seq_len):
        super().__init__()
        self.encoder = Encoder(model_dimension = model_dimension, num_layers=encoder_layers, hidden_layer_dimension=hidden_layer_dimension, key_dimension=key_dimension, value_dimension=value_dimension, num_heads = num_heads, vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.max_seq_len = max_seq_len
        self.classification_head = ClassificationHead(model_dimension, num_labels)

    def forward(self, tokens, masks):
        embeddings = self.encoder(tokens = tokens, masks = masks, sequence_length=self.max_seq_len)
        output = self.classification_head(embeddings) 
        return output
    
    def get_embeddings(self, tokens, masks):
        return self.encoder(tokens = tokens, masks = masks, sequence_length=self.max_seq_len)




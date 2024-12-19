import torch.nn as nn
import torch
from models.shared_architecture.classification_head import ClassificationHead

class Word2VecModel(nn.Module):
    def __init__(self, word2vec_model, model_dimension, num_labels):
        super().__init__()
        self.word2vec_model = word2vec_model
        self.classification_head = ClassificationHead(model_dimension, num_labels)

    def forward(self, tokens, masks):
        embeddings = []
        for sequence in tokens:
            int_seq = [int(token) for token in sequence]
            embeddings.append(torch.tensor(self.word2vec_model.wv[int_seq]))
        masks = (masks == 0).float()

        embeddings = torch.stack(embeddings)*masks.unsqueeze(-1)
        output = self.classification_head(embeddings)
        return output



import torch.nn as nn
import torch.nn.functional
import math
class ScaledDotProduct(nn.Module):
    def __init__(self, key_dimension):
        super().__init__()
        self.d_k = math.sqrt(key_dimension)

    
    def forward(self, queries, keys, values, masks):
        scores = torch.matmul(queries, keys.transpose(1,2)) / self.d_k
        unsqueezed_masks = masks.unsqueeze(1).expand(-1, scores.shape[2], -1)
        scores = scores + unsqueezed_masks
        softmaxed_scores = torch.softmax(scores, -1)
        output = torch.matmul(softmaxed_scores, values)

        return output

import torch
import torch.nn as nn
from models.transformer.scaled_dot_product import ScaledDotProduct

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, key_dimension, value_dimension, n_heads):
        super().__init__()
        self.model_dimension = model_dimension
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.heads = nn.ModuleList([])
        for i in range(n_heads):
            self.heads.append(ScaledDotProduct(key_dimension//n_heads))
        self.w_q = nn.Linear(model_dimension, key_dimension)
        self.w_k = nn.Linear(model_dimension, key_dimension)
        self.w_v = nn.Linear(model_dimension, value_dimension)

        self.final_output_layer = nn.Linear(value_dimension, model_dimension)
        


        
    def forward(self, queries, keys, values, masks):

        #initial dimensionality: batch_size x seq_length x query/key/value dimension
        batch_size = queries.size()[0]
        seq_length = queries.size()[1]

        #split the last dimension of queries/keys/value (batch, sequence, *model_dimension*) into two new dimensions: the number of heads and a dimension for the embeddings
        #which is the original model dimension divided by the number of heads

        # for every sentence in the batch, we do:
        # sentence_length x model_dimension * model_dimension x x_dimenions -> sentence_length x x_dimension
        #we transpose because we want the structure to be batch, head, sequence, embeddings
        #this is hierarchically intuitive, we want there to be 8 heads for a batch, a sequence (the same sequence) for each head
        # and then for each word in the sequence we want embeddings
        q = self.w_q(queries).view(batch_size, seq_length, len(self.heads), self.key_dimension//len(self.heads)).transpose(1,2)
        k = self.w_k(keys).view(batch_size, seq_length, len(self.heads), self.key_dimension//len(self.heads)).transpose(1,2)
        v = self.w_v(values).view(batch_size, seq_length, len(self.heads), self.value_dimension//len(self.heads)).transpose(1,2)
        attentions = []
        for i in range(len(self.heads)):
            #[:,i] gives us the contents of every ith head for every sentence in the batch ( represents every sentence (i.e. the batch dimension) and then i is for each head)
            attentions.append(self.heads[i](q[:,i], k[:,i],v[:,i],masks))    
            #dimensionality of everything being appended: seq_length * model_dimension/len(heads)

        #we now have to convert the attention embeddings that we have generate back into its original shape, so we switch the heads and sequence length again
        #and then reshape heads/embeddings for each head into just the embeddings, which is of the size model_dimension

        #final attention layer dimensionality: batch_size x seq_length x model_dimension
        attention = torch.cat(attentions,dim=-1).transpose(1,2).contiguous().view(batch_size,seq_length, self.value_dimension)
        return self.final_output_layer(attention)




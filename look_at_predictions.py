from utilities.data_processing_utils import preprocess_dataset
from models.transformer.ner_transformer_model import TransformerModel
import torch
from gensim.models import Word2Vec
from models.word2vec.word2vec_model import Word2VecModel
def look_at_predictions(data_path, test_sentences_path, model_path, output_path, model_dimension=512, heads=8, hidden_layer_dimension=0, key_dimension=0, value_dimension=0, encoder_layers=6, num_labels=9, word2vec_model_path=""):       
       dataset, length, dictionary =  preprocess_dataset(data_path, min_count=3)
       max_seq_len = max(len(sequence) for sequence in dataset['train'] ['tokens'])
       
       
       with open(test_sentences_path, 'r') as test_sentences_file:
              test_sentences = [test_sentence.split(" ") for test_sentence in test_sentences_file.read().split("\n")]
              tokenized_test_sentences =  [[dictionary[word.lower()] if word.lower() in dictionary else 1 for word in sentence] for sentence in test_sentences]
       
       seq_length = max(len(sequence) for sequence in tokenized_test_sentences)
       
       attention_masks=[]
       for sentence in tokenized_test_sentences:
              attention_masks.append([0]*len(sentence))
              for _ in range(len(sentence), max_seq_len):
                     sentence.append(0)
                     attention_masks[-1].append(-1e9)
       state_dict = torch.load(model_path)

       if len(word2vec_model_path) > 0:
              wv_model = Word2Vec.load(word2vec_model_path)
              model = Word2VecModel(wv_model, model_dimension=model_dimension, num_labels=9)
              model.load_state_dict(state_dict)
       else:
              hidden_layer_dimension = model_dimension * 2 if hidden_layer_dimension == 0 else hidden_layer_dimension
              key_dimension = model_dimension if key_dimension == 0 else key_dimension
              value_dimension = model_dimension if value_dimension == 0 else value_dimension
              model = TransformerModel(model_dimension=model_dimension, num_heads=heads, encoder_layers=encoder_layers, hidden_layer_dimension=hidden_layer_dimension, key_dimension=key_dimension, value_dimension=value_dimension, vocab_size=length, num_labels = num_labels, max_seq_len=max_seq_len)
              model.load_state_dict(state_dict)

       model.eval()

       
       to_write_to_file = ""
       preds_list = model.forward(masks=torch.tensor(attention_masks), tokens=torch.tensor(tokenized_test_sentences))
       for embeddings_sentence, test_sentence in zip(preds_list, test_sentences):
              for embedding, word in zip(embeddings_sentence, test_sentence):
                     to_write_to_file = to_write_to_file + word + ","
                     to_write_to_file = to_write_to_file + ",".join(str(value.item()) for value in embedding)
                     to_write_to_file = to_write_to_file + "\n"
              to_write_to_file = to_write_to_file + ",".join("-" for _ in range(len(embedding) + 1)) + "\n"
       with open(f"model_predictions/{output_path}", 'w') as f:
              f.write(to_write_to_file)
              


def main():
       look_at_predictions(data_path="tner/conll2003", test_sentences_path="texts/test_sentences.txt", model_path="model_outputs/2048_word2vec.pt", word2vec_model_path="model_outputs/2048_word2vec.pt_word2vec_embeddings.model", output_path="word2vec_embeddings.csv",model_dimension=2048, heads=0)
       look_at_predictions(data_path="tner/conll2003", test_sentences_path="texts/test_sentences.txt", model_path="model_outputs/transformer256_8_1epoch.pt", output_path="256_8_predictions_test.csv", model_dimension=256, heads=8)

if __name__ == "__main__":
    main()


       

       


       
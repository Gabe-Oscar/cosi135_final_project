import torch
from torch import nn as nn
from utilities.data_processing_utils import preprocess_dataset, get_pad_start
from utilities.metric_utils import StopWatch, generate_metrics
from models.transformer.ner_transformer_model import TransformerModel
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from gensim.models import Word2Vec
from models.word2vec.word2vec_model import Word2VecModel
from argparse import ArgumentParser
from utilities.output_to_spreadsheet import output_to_spreadsheet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_STRING = "[PAD]"
UNK_STRING = "[UNK]"
PAD = 0
UNK = 1
PAD_TAG = -1
MASK = -1e9
TOKEN = "tokens"
INDEX = "indices"

def parse_args():
   parser = ArgumentParser()
   
   # Architecture
   parser.add_argument('--model_dimension', type=int, default=256)
   parser.add_argument('--num_heads', type=int, default=8)
   parser.add_argument('--key_dimension', type=int, default=256) 
   parser.add_argument('--value_dimension', type=int, default=256)
   parser.add_argument('--hidden_dimension', type=int, default=512)
   parser.add_argument('--num_encoder_layers', type=int, default=6)
   parser.add_argument("--num_labels", type=int, default=9)

   # Training
   parser.add_argument('--batch_size', type=int, default=32)
   parser.add_argument('--epoch_count', type=int, default=10)
   parser.add_argument('--learning_rate', type=float, default=0.0005)
   parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

   # Data
   parser.add_argument('--dataset_name', type=str, default='tner/conll2003')
   parser.add_argument('--max_seq_len', type=int, default=None)

   # Model
   parser.add_argument('--model_type', type=str, choices=['transformer', 'word2vec'], default='transformer')
   
   # Output
   parser.add_argument("--model_results_path", default="")
   parser.add_argument("--model_path", default="")

   return parser.parse_args()



    



def train_epoch(model, dataloader, num_labels, loss_func, optimizer, stopWatch):
     model.train()
     total_loss = 0
     batch_count = 0
     for batch in dataloader:
          batch_count += 1
          
          loss = train_batch(model, batch, optimizer, loss_func, num_labels)
          total_loss += loss
          print(f"Batch {batch_count}, Loss: {loss}")
          stopWatch.print_elapsed_time()
     return total_loss / batch_count

def train_batch(model, batch, optimizer, loss_func, num_labels):
        tokens = batch['tokens']
        tags = batch['tags']
        attention_masks = batch['masks']
        tokens = tokens.to(device)
        attention_masks = attention_masks.to(device)
        tags = tags.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(tokens, attention_masks)
        loss = loss_func(outputs.view(-1,num_labels), tags.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

def validate(model, dataloader, loss_func, num_labels):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
            for batch in dataloader:
                 loss, flat_pred, flat_label = validate_batch(model, batch, loss_func, num_labels)
                 total_loss += loss
                 all_predictions.extend(flat_pred)
                 all_labels.extend(flat_label)       
            
    return total_loss/len(dataloader), all_predictions, all_labels 
            
def validate_batch(model, batch, loss_func, num_labels):     
    tokens = batch["tokens"]
    attention_mask = batch["masks"]
    tags = batch["tags"]


    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    tags = tags.to(device)

    outputs = model(tokens, attention_mask)

    batch_loss = loss_func(outputs.view(-1, num_labels), tags.view(-1))
    batch_predictions = torch.argmax(outputs, dim=-1)

    flattened_predictions = []
    flattened_labels = []

    for sequence_tags, sequence_preds in zip(tags, batch_predictions):
        sequence_tags = sequence_tags.cpu().numpy()
        sequence_preds = sequence_preds.cpu().numpy()
    
        
        pad_start = get_pad_start(sequence_tags)
        valid_preds = sequence_preds[:pad_start]
        valid_tags = sequence_tags[:pad_start]
        
        flattened_predictions.extend(valid_preds)
        flattened_labels.extend(valid_tags)
    
    return batch_loss.item(), flattened_predictions, flattened_labels
         


def train(train_dataloader, val_dataloader, loss_func, model, num_labels, optimizer, epoch_count):
    training_history = []
    stopWatch = StopWatch()
    for epoch in range(epoch_count):
        print(f"Epoch {epoch + 1}/{epoch_count}") 
        avg_train_loss = train_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer, loss_func=loss_func, num_labels=num_labels, stopWatch=stopWatch)  
        val_loss, predictions, labels = validate(model, val_dataloader, loss_func, num_labels)
        training_history.append(generate_metrics(avg_train_loss=avg_train_loss, val_loss=val_loss, predictions=predictions, labels=labels, epoch = epoch))
            
    elapsed_time = stopWatch.get_elapsed_time()
    return training_history[-1], elapsed_time



def load_training_data(dataset_name, batch_size, min_count, max_seq_len):
    
    def collate_function(data): 
        def pad(sequence, max_seq_len, pad):
            padded_sequence = sequence + [pad] * (max_seq_len-len(sequence))
            return padded_sequence 
        def cut_off(sequence, max_seq_len):
            return sequence[:max_seq_len]  
        token_list = []
        tag_list = []
        mask_list = []
        for pair in data:
            tokens = pair[0]
            tags = pair[1]
            attention_masks = [0] * len(tokens)
            token_list.append(cut_off(pad(tokens, max_seq_len, PAD), max_seq_len))
            tag_list.append(cut_off(pad(tags, max_seq_len,PAD_TAG), max_seq_len))
            mask_list.append(cut_off(pad(attention_masks, max_seq_len, MASK), max_seq_len))
        return {'tokens': torch.tensor(token_list), 'tags': torch.tensor(tag_list), 'masks': torch.tensor(mask_list)}

    indexicalized_dataset, vocab_size, index_dictionary = preprocess_dataset(dataset_name, min_count=min_count)
    train_dataloader = DataLoader(list(zip(indexicalized_dataset['train']['tokens'], indexicalized_dataset['train']['tags'])), batch_size = batch_size, shuffle = True, collate_fn=collate_function)
    val_dataloader = DataLoader(list(zip(indexicalized_dataset['validation']['tokens'], list(indexicalized_dataset['validation']['tags']))), batch_size = batch_size, shuffle = False, collate_fn=collate_function)
    return train_dataloader, val_dataloader, vocab_size, indexicalized_dataset, index_dictionary 
    


def main():
        args = parse_args()    
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        train_dataloader, val_dataloader, vocab_size, indexicalized_dataset, index_dictionary = load_training_data(args.dataset_name, args.batch_size, min_count = 3, max_seq_len=args.max_seq_len)


        if args.model_type == "transformer":
            model = TransformerModel(model_dimension=args.model_dimension, encoder_layers=args.num_encoder_layers, hidden_layer_dimension=args.hidden_dimension, key_dimension=args.key_dimension, value_dimension=args.value_dimension, num_heads = args.num_heads, vocab_size=vocab_size, num_labels=args.num_labels, max_seq_len=args.max_seq_len)
        elif args.model_type == "word2vec":       
            word2vec_model = Word2Vec(sentences = indexicalized_dataset['train']['tokens'], vector_size=args.model_dimension, epochs=20, window=5, sg=1,workers=4, min_count=3)
            word2vec_model.save(args.model_path + "_word2vec_embeddings.model")
            model = Word2VecModel(word2vec_model,model_dimension=args.model_dimension, num_labels=args.num_labels)
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        model.to(device)
        model.train()
        results, elapsed_time = train(train_dataloader, val_dataloader, loss_func=loss_func, model=model, num_labels=args.num_labels, epoch_count=args.epoch_count, optimizer=optimizer)
        if args.model_path != "": torch.save(model.state_dict(), args.model_path)
        if args.model_results_path != "":output_to_spreadsheet(output_filename=args.model_results_path, elapsed_time=elapsed_time, results=results, model_dimension=args.model_dimension, hidden_layer_dimension=args.hidden_dimension,num_encoder_layers=args.num_encoder_layers,num_heads=args.num_heads,key_dimension=args.key_dimension,value_dimension=args.value_dimension,batch_size=args.batch_size)
       
             

if __name__=="__main__":
    main()






             
         





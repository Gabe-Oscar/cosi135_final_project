import torch
from datasets import load_dataset
from collections import Counter

PAD_STRING = "[PAD]"
UNK_STRING = "[UNK]"
PAD = 0
UNK = 1
PAD_TAG = -1
MASK = -1e9
TOKEN = "tokens"
INDEX = "indices"

def get_pad_start(sequence_tags):
        try:
            return sequence_tags.tolist().index(PAD_TAG)
        except ValueError:
            return len(sequence_tags)



def preprocess_dataset(dataset_string, min_count=1):
   # def pad(example):
   #     sentence = example[TOKEN]
   #     sentence = sentence + [PAD] * (max_seq_len - len(sentence))
   #     example[TOKEN] = sentence
   #     return example
    
    def generate_index_dictionary(dataset):
        all_tokens = [token.lower() for sentence in dataset['train']['tokens'] for token in sentence]
        vocabulary = dict()
        counter = Counter(all_tokens)
        
        vocabulary[PAD] = 0
        vocabulary[UNK] = 1
        for token in all_tokens:
            if(token not in vocabulary and counter[token] >= min_count):
                vocabulary[token] = len(vocabulary)
        return vocabulary, counter

    def generate_indices(example):
        sentence = example[TOKEN]
        example[TOKEN] = torch.tensor([index_dictionary.get(word.lower() if counter[word.lower()] >= min_count else UNK, 1) for word in sentence])

        return example

        


    dataset = load_dataset(dataset_string)
    index_dictionary, counter = generate_index_dictionary(dataset)

    indexicalized_dataset = dataset.map(generate_indices)


    return indexicalized_dataset, len(index_dictionary), index_dictionary
               



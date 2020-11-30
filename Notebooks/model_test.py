import os
import jsonlines
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import Embedding
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy import inf
import matplotlib.pyplot as plt

def load_wiki(basedir, NUM_TOKEN, use_chars=False):
    datasets_fnames = {
        'train': os.path.join(basedir, 'en_train.jsonl'),
        'valid': os.path.join(basedir, 'en_valid.jsonl'),
        'test': os.path.join(basedir, 'en_test.jsonl'),
    }
    datasets_text = {
        'train': [],
        'valid': [],
        'test': [],
    }
    for split, fname in datasets_fnames.items():
        for token_dict in jsonlines.open(fname):
            # print(token_dict)
            if(use_chars):
                for i in range(NUM_TOKEN):
                    s = list(''.join(token_dict[i]['tokens']))
                    datasets_text[split].append(s)
            else:
                for i in range(NUM_TOKEN):
                    datasets_text[split].append(token_dict[i]['tokens'])
    return datasets_text

class Dictionary(object): #maps words to indices
    def __init__(self, datasets, include_valid=False):
        self.tokens = []
        self.ids = {}
        self.counts = {}

        # add special tokens
        self.add_token('<bos>') #beginning of sentence
        self.add_token('<eos>') #end of sentence
        self.add_token('<pad>')
        self.add_token('<unk>') #unknown. Needed in case use with text with word that isn't in vocab

        for line in tqdm(datasets['train']):
            for w in line:
                self.add_token(w)

        if include_valid is True:
            for line in tqdm(datasets['valid']):
                for w in line:
                    self.add_token(w)

    def add_token(self, w):
        if w not in self.tokens:
            self.tokens.append(w)
            _w_id = len(self.tokens) - 1
            self.ids[w] = _w_id
            self.counts[w] = 1
        else:
            self.counts[w] += 1

    def get_id(self, w):
        return self.ids[w]

    def get_token(self, idx):
        return self.tokens[idx]

    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]

    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids['<unk>'] for i in l]

    def __len__(self):
        return len(self.tokens)

def tokenize_dataset(datasets, dictionary, ngram_order=2):  # substitute words with numbers. Sometimes can include splitting strings, dealing with punctuation and symbols.
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        _current_dictified = []
        for l in tqdm(dataset):
            l = ['<bos>'] * (ngram_order - 1) + list(l) + ['<eos>']
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
    return tokenized_datasets

def pad_strings(minibatch):
    max_len_sample = max(len(i.split(' ')) for i in minibatch)
    result = []
    for line in minibatch:
        line_len = len(line.split(' '))
        padding_str = ' ' + '<pad> ' * (max_len_sample - line_len)
        result.append(line + padding_str)
    return result

class TensoredDataset():
    def __init__(self, list_of_lists_of_tokens):
        self.input_tensors = []
        self.target_tensors = []

        for sample in list_of_lists_of_tokens:
            self.input_tensors.append(torch.tensor([sample[:-1]], dtype=torch.long))
            self.target_tensors.append(torch.tensor([sample[1:]], dtype=torch.long))

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        # return a (input, target) tuple
        return (self.input_tensors[idx], self.target_tensors[idx])

def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padded_tensor = torch.cat([t, torch.tensor([[pad_token] * (max_length - t.size(-1))], dtype=torch.long)],
                                  dim=-1)
        padded_list.append(padded_tensor)

    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor

def pad_collate_fn(batch):
    input_list = [s[0] for s in batch]
    target_list = [s[1] for s in batch]
    pad_token = wiki_dict.get_id('<pad>')
    input_tensor = pad_list_of_tensors(input_list, pad_token)
    target_tensor = pad_list_of_tensors(target_list, pad_token)
    return input_tensor, target_tensor

class LSTMLanguageModel(nn.Module):
    """
    This model combines embedding, lstm and projection layer into a single model
    """
    def __init__(self, options):
        super().__init__()

        self.lookup = nn.Embedding(num_embeddings=options['num_embeddings'], embedding_dim=options['embedding_dim'], padding_idx=options['padding_idx'])
        self.lstm = nn.LSTM(options['input_size'], options['hidden_size'], options['num_layers'], dropout=options['lstm_dropout'], batch_first=True)
        self.projection = nn.Linear(options['hidden_size'], options['num_embeddings'])

    def forward(self, encoded_input_sequence):
        """
        Forward method process the input from token ids to logits
        """
        embeddings = self.lookup(encoded_input_sequence)
        lstm_outputs = self.lstm(embeddings)
        logits = self.projection(lstm_outputs[0])

        return logits

def model_training(model, optimizer, num_epochs):
  plot_cache = []
  best_loss = float(inf)
  no_improvement = 0

  for epoch_number in range(num_epochs):
      avg_loss=0
      model.train()
      train_log_cache = []
      for i, (inp, target) in enumerate(wiki_loaders['train']):
          optimizer.zero_grad()
          inp = inp.to(current_device)
          target = target.to(current_device)
          logits = model(inp)
          loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
          loss.backward()
          optimizer.step()
          train_log_cache.append(loss.item())
      avg_loss = sum(train_log_cache)/len(train_log_cache)
      print('Training loss after {} epoch = {:.{prec}f}'.format(epoch_number, avg_loss, prec=4))

      valid_losses = []
      model.eval()
      with torch.no_grad():
        for i, (inp, target) in enumerate(wiki_loaders['valid']):
            inp = inp.to(current_device)
            target = target.to(current_device)
            logits = model(inp)

            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            valid_losses.append(loss.item())
        avg_val_loss = sum(valid_losses) / len(valid_losses)
        print('Validation loss after {} epoch = {:.{prec}f}'.format(epoch_number, avg_val_loss, prec=4))

        if (avg_val_loss < best_loss):
          best_loss = avg_val_loss
        else:
          no_improvement += 1

        if(no_improvement >= 5):
          print('Early stopping at epoch: %d', epoch_number)
          break
      plot_cache.append((avg_loss, avg_val_loss))

  return plot_cache, best_loss

if __name__ == '__main__':
    # Usage: python model_test.py CHAR/WORD NUM_EPOCHS NUM_TOKEN
    import sys
    USE_CHARS = True if sys.argv[1]=='CHAR' else None
    USE_CHARS = False if sys.argv[1]=='WORD' else USE_CHARS
    NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv)>2 else 100
    NUM_TOKEN = int(sys.argv[3]) if len(sys.argv)>3 else 10000
    BATCH_SIZE = int(sys.argv[4]) if len(sys.argv)>4 else 128

    type = 'char' if USE_CHARS == True else 'word'
    print('model type:', type)

    wiki_loaders = {}

    batch_size = BATCH_SIZE
    print('batch size:', batch_size)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        current_device = 'cuda'
    else:
        current_device = 'cpu'
    print('device:', current_device)

    # print('start loading data')
    wiki_dataset = load_wiki('./data/en_json/', use_chars=USE_CHARS, NUM_TOKEN=NUM_TOKEN)
    # print('done loading data')
    wiki_dict = Dictionary(wiki_dataset, include_valid=True)

    # print('start tokenizing')
    wiki_tokenized_datasets = tokenize_dataset(wiki_dataset, wiki_dict)
    # print('done tokenizing')
    wiki_tensor_dataset = {}

    for split, listoflists in wiki_tokenized_datasets.items():
        wiki_tensor_dataset[split] = TensoredDataset(listoflists)

    for split, wiki_dataset in wiki_tensor_dataset.items():
        wiki_loaders[split] = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    embedding_size = 256
    hidden_size = 1024
    num_layers = 3
    lstm_dropout = 0.3

    options = {
        'num_embeddings': len(wiki_dict),
        'embedding_dim': embedding_size,
        'padding_idx': wiki_dict.get_id('<pad>'),
        'input_size': embedding_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'lstm_dropout': lstm_dropout,
    }
    print(options)

    LSTM_model_en = LSTMLanguageModel(options).to(current_device)

    criterion = nn.CrossEntropyLoss(ignore_index=wiki_dict.get_id('<pad>'))

    model_parameters = [p for p in LSTM_model_en.parameters() if p.requires_grad]
    optimizer = optim.SGD(model_parameters, lr=0.001, momentum=0.999)
    filename = './saved_models/LSTM_en_'+type+'_'+str(NUM_TOKEN)+'tklen_'+str(BATCH_SIZE)+'bsize_'+str(NUM_EPOCHS)+'ep.pt'
    plot_en, loss = model_training(model=LSTM_model_en, optimizer=optimizer, num_epochs=NUM_EPOCHS)
    torch.save({'model_state_dict': LSTM_model_en.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'plot_cache': plot_en,
                'loss': loss,
                }, filename)
    print(filename)

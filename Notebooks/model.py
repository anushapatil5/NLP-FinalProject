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
import pickle
from generate_pickle import Dictionary
import time

def load_pickle(path):
    with open(path, 'rb') as handle:
        tokenized_datasets = pickle.load(handle)
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
    pad_token = 2 # wiki_dict.get_id('<pad>')
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
      start_time = time.time()
      for i, (inp, target) in enumerate(wiki_loaders['train']):
          optimizer.zero_grad()
          inp = inp.to(current_device)
          target = target.to(current_device)
          logits = model(inp)
          loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
          loss.backward()
          optimizer.step()
          train_log_cache.append(loss.item())
      if current_device == 'cuda':
          print(torch.cuda.get_device_name(0))
          print('Memory Usage:')
          print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
          print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
      avg_loss = sum(train_log_cache)/len(train_log_cache)
      torch.cuda.empty_cache()
      print('Training loss after {} epoch = {:.{prec}f}'.format(epoch_number+1, avg_loss, prec=4))
      print(time.time()-start_time)

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
        torch.cuda.empty_cache()
        print('Validation loss after {} epoch = {:.{prec}f}'.format(epoch_number+1, avg_val_loss, prec=4))

        if (avg_val_loss < best_loss):
          best_loss = avg_val_loss
        else:
          no_improvement += 1

        if(no_improvement >= 5):
          print('Early stopping at epoch: %d', epoch_number+1)
          break
      plot_cache.append((avg_loss, avg_val_loss))

  return plot_cache, best_loss

if __name__ == '__main__':
    # Usage: python model_test.py [LANG] [TYPE] [NUM_EPOCHS] [BATCH_SIZE]
    # LANG (str): ar, en, it, hi
    # TYPE (str): CHAR or WORD
    # NUM_EPOCHS (int): number of epochs to train for
    # BATCH_SIZE (int): batch size

    import sys

    LANG = sys.argv[1]
    USE_CHARS = True if sys.argv[2]=='CHAR' else None
    USE_CHARS = False if sys.argv[2]=='WORD' else USE_CHARS
    NUM_EPOCHS = int(sys.argv[3]) if len(sys.argv)>3 else 100
    BATCH_SIZE = int(sys.argv[4]) if len(sys.argv)>4 else 128
    type = 'char' if USE_CHARS == True else 'word'

    print('model language and type:', LANG, type)

    batch_size = BATCH_SIZE
    print('batch size:', batch_size)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        current_device = 'cuda'
    else:
        current_device = 'cpu'
    print('device:', current_device)
    if current_device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    ############################################################################
    ######################### AFTER GENERATING PICKLE ##########################
    ############################################################################
    PATH = LANG+'_'+type+'_tokenized.pickle'

    wiki_loaders = {}

    print('start loading')
    wiki_tokenized_datasets = load_pickle(path=PATH)
    print('done loading')

    wiki_path = LANG+'_'+type+'_wiki_dict_filtered.pickle'
    with open(wiki_path, 'rb') as handle:
        wiki_dict = pickle.load(handle)
    print(len(wiki_dict.ids))

    wiki_tensor_dataset = {}

    for split, listoflists in wiki_tokenized_datasets.items():
        wiki_tensor_dataset[split] = TensoredDataset(listoflists)

    for split, wiki_dataset in wiki_tensor_dataset.items():
        wiki_loaders[split] = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    embedding_size = int(256)
    hidden_size = int(1024)
    num_layers = 2
    lstm_dropout = 0.3
    if USE_CHARS:
        num_embeddings = len(wiki_dict.ids)
    if (not USE_CHARS):
        num_embeddings = len(wiki_dict.ids)

    options = {
        'num_embeddings': num_embeddings,
        'embedding_dim': embedding_size,
        'padding_idx': 2, #wiki_dict.get_id('<pad>')
        'input_size': embedding_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'lstm_dropout': lstm_dropout,
    }
    print(options)

    LSTM_model = LSTMLanguageModel(options).to(current_device)

    criterion = nn.CrossEntropyLoss(ignore_index=2) #wiki_dict.get_id('<pad>')

    model_parameters = [p for p in LSTM_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model_parameters, lr=0.001, momentum=0.999)
    filename = './saved_models/LSTM_'+LANG+'_'+type+'_'+str(BATCH_SIZE)+'bsize_'+str(embedding_size)+'emb_'+str(hidden_size)+'hdim_'+str(num_layers)+'lyrs_'+str(NUM_EPOCHS)+'ep.pt'
    plot, loss = model_training(model=LSTM_model, optimizer=optimizer, num_epochs=NUM_EPOCHS)
    torch.save({'model_state_dict': LSTM_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'plot_cache': plot,
                'loss': loss,
                }, filename)
    print(filename)

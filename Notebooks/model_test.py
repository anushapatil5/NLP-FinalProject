import os
import jsonlines
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy
import matplotlib.pyplot as plt


def load_personachat(basedir, use_chars=False):
    datasets_fnames = {
        'train': os.path.join(basedir, 'personachat_all_sentences_train.jsonl'),
        'valid': os.path.join(basedir, 'personachat_all_sentences_valid.jsonl'),
        'test': os.path.join(basedir, 'personachat_all_sentences_test.jsonl'),
    }
    datasets_text = {
        'train': [],
        'valid': [],
        'test': [],
    }
    for split, fname in datasets_fnames.items():
        for token_dict in jsonlines.open(fname):
            if(use_chars):
                s = list(''.join(token_dict['tokens']))
                datasets_text[split].append(s)
            else:
                datasets_text[split].append(token_dict['tokens'])
    return datasets_text


class Dictionary(object):
    def __init__(self, datasets, include_valid=False):
        self.tokens = []
        self.ids = {}
        self.counts = {}
        self.add_token('<bos>')
        self.add_token('<eos>')
        self.add_token('<pad>')
        self.add_token('<unk>')

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


def pad_strings(minibatch):
    max_len_sample = max(len(i.split(' ')) for i in minibatch)
    result = []
    for line in minibatch:
        line_len = len(line.split(' '))
        padding_str = ' ' + '<pad> ' * (max_len_sample - line_len)
        result.append(line + padding_str)
    return result


def tokenize_dataset(datasets, dictionary, ngram_order=2):  # substitute words with numbers. Sometimes can include splitting strings, dealing with punctuation and symbols.
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        _current_dictified = []
        for l in tqdm(dataset):
            l = ['<bos>'] * (ngram_order - 1) + l + ['<eos>']
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified

    return tokenized_datasets


class TensoredDataset(Dataset):
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


USE_CHARS = False

# TODO: Check if we are taking all characters.
# TODO: Make another model for LSTM.
# TODO: Save states for model by loangauge and model type
# TODO: General checkup.

personachat_dataset = load_personachat('personachat/', use_chars=USE_CHARS)
persona_dict = Dictionary(personachat_dataset, include_valid=True)
print("Hello")

personachat_tokenized_datasets = tokenize_dataset(personachat_dataset, persona_dict)
persona_tensor_dataset = {}

# TODO What is this?
for split, listoflists in personachat_tokenized_datasets.items():
    persona_tensor_dataset[split] = TensoredDataset(listoflists)




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
    pad_token = persona_dict.get_id('<pad>')
    input_tensor = pad_list_of_tensors(input_list, pad_token)
    target_tensor = pad_list_of_tensors(target_list, pad_token)
    return input_tensor, target_tensor


persona_loaders = {}

batch_size = 128

for split, persona_dataset in persona_tensor_dataset.items():
    persona_loaders[split] = DataLoader(persona_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)


print("TEST")


class RNNLanguageModel(nn.Module):  # RNN is only difference from previous model
    """
    This model combines embedding, rnn and projection layer into a single model
    """

    def __init__(self, options):
        super().__init__()

        # create each LM part here
        self.lookup = nn.Embedding(num_embeddings=options['num_embeddings'], embedding_dim=options['embedding_dim'],
                                   padding_idx=options['padding_idx'])
        self.rnn = nn.RNN(options['input_size'], options['hidden_size'], options['num_layers'],
                          dropout=options['rnn_dropout'], batch_first=True)
        self.projection = nn.Linear(options['hidden_size'], options['num_embeddings'])

    def forward(self, encoded_input_sequence):
        """
        Forward method process the input from token ids to logits
        """
        embeddings = self.lookup(encoded_input_sequence)
        rnn_outputs = self.rnn(embeddings)
        logits = self.projection(
            rnn_outputs[0])  # convenient for seq to seq models. check shape of output. lstm gives different

        return logits


load_pretrained = False

num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    current_device = 'cuda'
else:
    current_device = 'cpu'

if load_pretrained:
    if not os.path.exists('personachat_rnn_lm.pt'):
        raise EOFError('Download pretrained model!')
    model_dict = torch.load('personachat_rnn_lm.pt', map_location=current_device)

    options = model_dict['options']
    model = RNNLanguageModel(options).to(current_device)
    model.load_state_dict(model_dict['model_dict'])

else:
    embedding_size = 256
    hidden_size = 512
    num_layers = 3
    rnn_dropout = 0.3

    options = {
        'num_embeddings': len(persona_dict),
        'embedding_dim': embedding_size,
        'padding_idx': persona_dict.get_id('<pad>'),
        'input_size': embedding_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'rnn_dropout': rnn_dropout,
    }

    model = RNNLanguageModel(options).to(current_device)

criterion = nn.CrossEntropyLoss(ignore_index=persona_dict.get_id('<pad>'))

model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(model_parameters, lr=0.001, momentum=0.999)

print("MODEL: ", model)

plot_cache = []

for epoch_number in range(100):
    avg_loss = 0
    if not load_pretrained:
        # do train
        model.train()
        train_log_cache = []
        for i, (inp, target) in tqdm(enumerate(persona_loaders['train'])):
            optimizer.zero_grad()
            inp = inp.to(current_device)
            target = target.to(current_device)
            logits = model(inp)

            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            loss.backward()
            optimizer.step()

            train_log_cache.append(loss.item())

            if i % 100 == 0:
                avg_loss = sum(train_log_cache) / len(train_log_cache)
                print('Step {} avg train loss = {:.{prec}f}'.format(i, avg_loss, prec=4))
                train_log_cache = []

    # do valid
    valid_losses = []
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(persona_loaders['valid']):
            inp = inp.to(current_device)
            target = target.to(current_device)
            logits = model(inp)

            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            valid_losses.append(loss.item())
        avg_val_loss = sum(valid_losses) / len(valid_losses)
        print('Validation loss after {} epoch = {:.{prec}f}'.format(epoch_number, avg_val_loss, prec=4))

    plot_cache.append((avg_loss, avg_val_loss))

    if load_pretrained:
        break
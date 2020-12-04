import os
import jsonlines
from tqdm import tqdm

import pickle

def load_wiki(basedir, LANG, use_chars=False):
    datasets_fnames = {
        'train': os.path.join(basedir, LANG+'_json', LANG+'_train.jsonl'),
        'valid': os.path.join(basedir, LANG+'_json', LANG+'_valid.jsonl'),
        'test': os.path.join(basedir, LANG+'_json', LANG+'_test.jsonl'),
    }
    datasets_text = {
        'train': [],
        'valid': [],
        'test': [],
    }
    for split, fname in datasets_fnames.items():
        for token_dict in jsonlines.open(fname):
            if(use_chars):
                for i in range(len(token_dict)):
                    s = list(''.join(token_dict[i]['tokens']))
                    datasets_text[split].append(s)
            else:
                for i in range(len(token_dict)):
                    datasets_text[split].append(token_dict[i]['tokens'])
    type = 'char' if USE_CHARS == True else 'word'
    filename = LANG+'_'+type+'_datasets_text.pickle'
    print('datasets_text filename:',filename)
    with open(filename, 'wb') as handle:
        pickle.dump(datasets_text, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return datasets_text, filename

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

def tokenize_dataset(path, dictionary, ngram_order=2):  # substitute words with numbers. Sometimes can include splitting strings, dealing with punctuation and symbols.
    with open(path, 'rb') as handle:
        datasets = pickle.load(handle)
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        _current_dictified = []
        for l in tqdm(dataset):
            l = ['<bos>'] * (ngram_order - 1) + list(l) + ['<eos>']
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
    filename = LANG+'_'+type+'_tokenized.pickle'
    print('tokenized_datasets filename:',filename)
    with open(filename, 'wb') as handle:
        pickle.dump(tokenized_datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenized_datasets

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

if __name__ == '__main__':
    # Usage: python model_test.py [LANG] [TYPE]
    # LANG (str): ar, en, it, hi
    # TYPE (str): CHAR or WORD

    import sys
    LANG = sys.argv[1]
    USE_CHARS = True if sys.argv[2]=='CHAR' else None
    USE_CHARS = False if sys.argv[2]=='WORD' else USE_CHARS
    type = 'char' if USE_CHARS == True else 'word'

    print('language and type:', LANG, type)

    print('start loading data')
    wiki_dataset, path = load_wiki('./data/', LANG=LANG, use_chars=USE_CHARS)
    # # Use the code below to avoid re-downloading the .json data
    # path = LANG+'_'+type+'_datasets_text.pickle'
    # with open(path, 'rb') as handle:
    #     wiki_dataset = pickle.load(handle)
    print(wiki_dataset['train'][0])
    print('done loading data')

    print('saved wiki dataset path:',path)

    print('start creating dictionary class')
    wiki_dict = Dictionary(wiki_dataset, include_valid=True)
    wiki_path = path[:7]
    print(wiki_path+'_wiki_dict.pickle')
    with open(wiki_path+'_wiki_dict.pickle', 'wb') as handle:
        pickle.dump(wiki_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Use the code below to download the pickled dict
    # filename = wiki_path+'_wiki_dict.pickle'
    # with open(filename, 'rb') as handle:
    #     wiki_dict = pickle.load(handle)
    print(len(wiki_dict.ids))
    print('done creating dictionary class')

    if USE_CHARS:
        sorted_counts = {k: v for k, v in sorted(wiki_dict.counts.items(), key=lambda item: item[1])}
        if LANG == 'en':
            res = {k: v for k, v in sorted_counts.items() if isEnglish(k)}
        else:
            res = {k: v for k, v in sorted_counts.items() if v > 1000}

        # convert tokens into unk and change their ids to unk's id.
        print('len of dict before filtering:',len(wiki_dict.ids))
        new_wiki_dict = {}
        new_tokens = ['<bos>', '<eos>','<pad>','<unk>']

        # add special tokens
        new_wiki_dict['<bos>'] = 0
        new_wiki_dict['<eos>'] = 1
        new_wiki_dict['<pad>'] = 2
        new_wiki_dict['<unk>'] = 3
        for token in wiki_dict.tokens:
          if token in res:
            new_wiki_dict[token] = wiki_dict.ids[token]
            new_tokens.append(token)

        wiki_dict.ids = new_wiki_dict
        wiki_dict.tokens = new_tokens
        for i, (key, val) in enumerate(wiki_dict.ids.items()):
            wiki_dict.ids[key] = i

        temp_count = {}
        for key, val in wiki_dict.counts.items():
            if key in wiki_dict.ids:
                temp_count[key] = val
        wiki_dict.counts = temp_count
        print('len of dict after filtering:',len(wiki_dict.ids))

    else:
        sorted_counts_word = {k: v for k, v in sorted(wiki_dict.counts.items(), key=lambda item: item[1])}
        # TODO: figure out how to filter words that do not occur frequently
        res_word = {k: v for k, v in sorted_counts_word.items() if v >= 10}

        # convert tokens into unk and change their ids to unk's id.
        print('len of dict before filtering:',len(wiki_dict.ids))
        new_wiki_dict = {}
        new_tokens = ['<bos>', '<eos>','<pad>','<unk>']

        # add special tokens
        new_wiki_dict['<bos>'] = 0
        new_wiki_dict['<eos>'] = 1
        new_wiki_dict['<pad>'] = 2
        new_wiki_dict['<unk>'] = 3
        for token in wiki_dict.tokens:
          if token in res_word:
            new_wiki_dict[token] = wiki_dict.ids[token]
            new_tokens.append(token)

        wiki_dict.ids = new_wiki_dict
        wiki_dict.tokens = new_tokens
        for i, (key, val) in enumerate(wiki_dict.ids.items()):
            wiki_dict.ids[key] = i

        temp_count = {}
        for key, val in wiki_dict.counts.items():
            if key in wiki_dict.ids:
                temp_count[key] = val
        wiki_dict.counts = temp_count
        print('len of dict after filtering:',len(wiki_dict.ids))

    # Use the code below to download the pickled dict
    filename = wiki_path+'_wiki_dict_filtered.pickle'
    # Store dictionary for future use
    with open(filename, 'wb') as handle:
        pickle.dump(wiki_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(filename, 'rb') as handle:
    #     wiki_dict = pickle.load(handle)
    print('start tokenizing')
    wiki_tokenized_datasets = tokenize_dataset(path, wiki_dict)
    print('done tokenizing')

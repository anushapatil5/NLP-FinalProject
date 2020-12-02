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
                # for i in range(10000):
                    s = list(''.join(token_dict[i]['tokens']))
                    datasets_text[split].append(s)
            else:
                for i in range(len(token_dict)):
                # for i in range(10000):
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
    print('done loading data')

    print(path)

    wiki_path = path[:7]
    wiki_dict = Dictionary(wiki_dataset, include_valid=True)

    if USE_CHARS:
        sorted_counts = {k: v for k, v in sorted(wiki_dict.counts.items(), key=lambda item: item[1])}
        if LANG == 'en':
            res = {k: v for k, v in sorted_counts.items() if isEnglish(k)}
        else:
            res = {k: v for k, v in sorted_counts.items() if v > 1000}
        print(len(res))

        # convert tokens into unk and change their ids to unk's id.
        print('len of dict before filtering:',len(wiki_dict))
        for token in wiki_dict.tokens:
          if token not in res:
            id = wiki_dict.get_id(token)
            print(token, id)
            wiki_dict.tokens[id] = '<unk>'
            wiki_dict.ids[token] = 3

        print('len of dict after filtering:',len(res))

    else:
        sorted_counts_word = {k: v for k, v in sorted(wiki_dict.counts.items(), key=lambda item: item[1])}
        res_word = {k: v for k, v in sorted_counts_word.items() if v > 100}

        # convert tokens into unk and change their ids to unk's id.
        print('len of dict before filtering:',len(wiki_dict))
        for token in wiki_dict.tokens:
          if token not in res_word:
            id = wiki_dict.get_id(token)
            print(token, id)
            wiki_dict.tokens[id] = '<unk>'
            wiki_dict.ids[token] = 3

        print('len of dict after filtering:',len(res_word))

    print('start tokenizing')
    wiki_tokenized_datasets = tokenize_dataset(path, wiki_dict)
    print('done tokenizing')

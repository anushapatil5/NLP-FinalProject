import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_chars(path):
    with open(path, 'r') as f:
        text = f.read()
    return tuple(set(text))


class CharDictionary(object):
    def __init__(self, path):
        self.chars = get_chars(path)
        self.idx2char = list(self.chars)
        self.char2idx = {ch: ii for ii, ch in enumerate(self.idx2char)}

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary_c = CharDictionary(os.path.join(path, 'train.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.char_train = self.dictionary_c

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding='utf-8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corpus = Corpus('./data/wikitext-2')

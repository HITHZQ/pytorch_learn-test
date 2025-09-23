from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

class LangEmbed:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0 : "SOS", 1: "EOS" }
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

MAX_LENGTH = 10
eng_prefixes = ("i am", "i m", "he is", "he s", "she is", "she s", "you are", "you are", "you are", "you re", "you are", "you are", "we are", "we re", "they are", "they re")

def filterPairs(pairs):
    p = []
    for pair in pairs:
        if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[0].startswith(eng_prefixes[0]):
            p.append(pair)
    return p

def prepareDate():
    lines = open(''
                 '')
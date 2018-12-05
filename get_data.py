import pprint
pp=pprint.PrettyPrinter(indent='3')
import os
import sys
sys.path.append('..')
import numpy as np
from data import munroecorpus
import torch
import torchtext.vocab as vocab


#Vocabulary
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: "PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addDescription(self, description):
        for word in description.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def make_list() :
    training = munroecorpus.get_training_handles()
    names = list(training[0].keys())
    return names

def hsv_to_rgb(h,s,v) :
    C = np.multiply(v, s)
    X = C * (1 - (np.abs(np.mod(np.multiply(h, 1 / 60), 2) - 1)))
    m = v - C
    r_,g_,b_ = [],[],[]

    for i, h in enumerate(h):

        if h < 60:
            r, g, b = C[i], X[i], 0
        elif h < 120:
            r, g, b = X[i], C[i], 0
        elif h < 180:
            r, g, b = 0, C[i], X[i]
        elif h < 240:
            r, g, b = 0, X[i], C[i]
        elif h < 300:
            r, g, b = X[i], 0, C[i]
        elif h < 360:
            r, g, b = C[i], 0, X[i]
        r_.append(r + m[i])
        g_.append(g + m[i])
        b_.append(b + m[i])
        rgb = []
    for i in range(len(r_)) :
        rgb.append( [ r_[i],g_[i],b_[i] ])
    return rgb

def make_pairs(colors, mode) :
    vocabulary = Vocabulary('eng')
    print("Read %s descriptions" % len(colors))
    RGB = []
    if mode == 'train' :
        for i,description in enumerate(colors) :
            vocabulary.addDescription(description)
            #HSV values between 0 and 1
            h = munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=0))
            s = np.multiply( munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=1)) , 0.01)
            v = np.multiply(munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=2)),0.01)
            RGB.append(hsv_to_rgb(h,s,v))
            colors[i] = description.replace('-',' ')
    elif mode == 'test' :
        #in hsv,
        for i,description in enumerate(colors) :
            vocabulary.addDescription(description)
            #HSV values between 0 and 1
            hsv = munroecorpus.open_datafile(munroecorpus.get_test_filename(description))
            h = [ _[0] for _ in hsv]
            s = np.multiply([ _[1] for _ in hsv],0.01)
            v = np.multiply([ _[2] for _ in hsv],0.01)
            RGB.append(hsv_to_rgb(h,s,v))
            colors[i] = description.replace('-',' ')
    print("Counted words:")
    print(vocabulary.n_words)
    pairs = list(zip(colors,colors))
    RGB = dict(zip(colors, RGB))
    return pairs, vocabulary, RGB

def get_embeddings (vocabulary,embedding_space):
    s = vocab.GloVe('6B')
    sample = s.vectors[0]

    #random uniform(Min,Max) initialization of unknown words, else init. with gloVe embeddings
    r1 = torch.max(s.vectors)
    r2 = torch.min(s.vectors)
    embeddings  = np.zeros([ vocabulary.n_words, embedding_space])
    number_unknown = 0
    for i in range(vocabulary.n_words) :
        word = vocabulary.index2word[i]
        if word in s.stoi :
            embeddings[i] = s.vectors[s.stoi[word]]
        else :
            number_unknown += 1
            embeddings[i] = (r1 - r2) * torch.rand_like(sample) + r2
    return torch.from_numpy(embeddings).float()

def load_models(diagonal = False) :
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'  # '/Users/sami/Desktop/project/'
    path = abs_path + 'dec'
    if diagonal: path += '_s'
    dec = torch.load(path, map_location='cpu')
    path = abs_path + 'enc'
    if diagonal: path += '_s'
    enc = torch.load(path, map_location='cpu')
    path = abs_path + 'lin'
    if diagonal: path += '_s'
    lin = torch.load(path, map_location='cpu')
    return enc,lin,dec

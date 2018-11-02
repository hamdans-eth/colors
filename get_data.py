import pprint
pp=pprint.PrettyPrinter(indent='3')
import numpy as np
import sys
sys.path.append('..')

from data import munroecorpus


#Vocabulary
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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




#TODO full dataset (right now just train)
def make_list( ) :
    training = munroecorpus.get_training_handles()
    names = list(training[0].keys())
    #deal with -

    return names


def make_pairs(colors) :
    vocabulary = Vocabulary('eng')
    print("Read %s descriptions" % len(colors))
    for description in colors :
        vocabulary.addDescription(description)
    print("Counted words:")
    print(vocabulary.n_words)
    names = [words.split() for words in colors]
    pairs = list(zip(names,names))
    return pairs, vocabulary

def make_pairs2(colors) :
    vocabulary = Vocabulary('eng')
    print("Read %s descriptions" % len(colors))
    for description in colors :
        vocabulary.addDescription(description)
    print("Counted words:")
    print(vocabulary.n_words)
    pairs = list(zip(colors,colors))
    return pairs, vocabulary

#training only
def make_pairs3(colors, mode) :
    vocabulary = Vocabulary('eng')
    print("Read %s descriptions" % len(colors))
    RGB = []
    if mode == 'training' :
        for i,description in enumerate(colors) :
            vocabulary.addDescription(description)
            #HSV values between 0 and 1
            h = munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=0))
            s = munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=1))
            v = munroecorpus.open_datafile(munroecorpus.get_training_filename(description, dim=2))
            h = h * 360
            C = v * s
            X = C * (1 - (np.abs(h/60 % 2 -1)) )
            m = v - C
            switch 
            colors[i] = description.replace('-',' ')
    elif mode == 'test' : print('ok')
    print("Counted words:")
    print(vocabulary.n_words)
    pairs = list(zip(colors,colors))
    return pairs, vocabulary
mode = 'training'
make_pairs3(make_list(),mode)


# #get  lists of full paths to data files
# colorname = 'acid green'
# training = munroecorpus.get_training_handles()
# dev = munroecorpus.get_dev_handles()
# test = munroecorpus.get_test_handles()
# train_fn = munroecorpus.get_training_filename(colorname,dim = 0) #supply dim = 0,1,2 for h,s,v
# dev_fn = munroecorpus.get_dev_filename(colorname)
# test_fn = munroecorpus.get_test_filename(colorname)
#
#
# example_train_data = munroecorpus.open_datafile(train_fn)
# example_dev_data = munroecorpus.open_datafile(dev_fn)
# example_test_data = munroecorpus.open_datafile(test_fn)
#
# names = list(training[0].keys())
#
# print(make_pairs(names))




###
# print("rgb: \n%s" % pp.pformat(example_test_data))
# #
# print("Example Training Handles:\n %s" % pp.pformat(list(training[0:3])[:10])) #pp.pformat(list(training[0:3].items())[:10]))
# print("Example Dev Handles:\n %s" % pp.pformat(list(dev.items())[:10]))
# print("Example Test Handles:\n %s" % pp.pformat(list(test.items())[:10]))
#
# print("Example Training Filename:\n %s" % pp.pformat(train_fn))
# print("Example Dev Filename :\n %s" % pp.pformat(dev_fn))
# print("Example Test Filename:\n %s" % pp.pformat(test_fn))
#
# print("Example Training Data: \n%s" % pp.pformat(example_train_data[:10]))
# print("Example Dev Data: \n%s" % pp.pformat(example_dev_data[:10]))
# print("Example Test Data: \n%s" % pp.pformat(example_test_data[:10]))
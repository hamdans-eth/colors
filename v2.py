from __future__ import unicode_literals, print_function, division
from get_data import make_list, make_pairs2

import rnn
import random
import torch.nn as nn
from torch import optim
import torch
import torchtext.vocab as vocab
import numpy as np
import time
import math

random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Constants
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 4
USE_ATTN = False
hidden_size = 3
teacher_forcing_ratio = 0 #no need
mu = 1 # for enforcing RGB distance


#Getting data
colors= make_list()

pairs,vocabulary = make_pairs2(colors)

#get weights
s = vocab.GloVe('6B')
sample = s.vectors[0]
 # assuming sample is known
embeddings_dim = s.dim #300


#random uniform initialization of unknown words, else init. with gloVe embeddings
r1 = torch.max(s.vectors)
r2 = torch.min(s.vectors)
embeddings  = np.zeros([ vocabulary.n_words, embeddings_dim ])
number_unknown = 0
for i in range(vocabulary.n_words) :
    word = vocabulary.index2word[i]
    if word in s.stoi :
        embeddings[i] = s.vectors[s.stoi[word]]
    else :
        number_unknown += 1
        embeddings[i] = (r1 - r2) * torch.rand_like(sample) + r2
embeddings = torch.from_numpy(embeddings).float()




def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #Get hidden state
    encoder_hidden = encoder.initHidden()
    #Clears the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        print('encoder out size')
        print(encoder_output.size())
    #start of seq
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    #teacher forcing for faster training !
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if(USE_ATTN) :
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else :
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #TODO
            if (USE_ATTN):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


## utility functions

def indexesFromDescription(vocabulary, description):
    return [vocabulary.word2index[word] for word in description.split(' ')]

def tensorFromDescription(vocabulary, description):
    indexes = indexesFromDescription(vocabulary, description)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromDescription(vocabulary, pair[0])
    target_tensor = tensorFromDescription(vocabulary, pair[1])
    return (input_tensor, target_tensor)



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    #penalizes the wrong probability
    #TODO implement the RGB distance to training RGB
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]
        #setting output size = input size => what about padding
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]


        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        #TODO : Plots ?
        #showPlot(plot_losses)


#from tutorial
def evaluate(encoder, decoder, description, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromDescription(vocabulary, description)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)

            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocabulary.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



#
encoder1 = rnn.EncoderRNN(vocabulary.n_words, hidden_size,embeddings).to(device)
if(not USE_ATTN) :
    decoder1 = rnn.DecoderRNN(hidden_size,embeddings,vocabulary.n_words).to(device)
else :
    decoder1 = rnn.AttnDecoderRNN(hidden_size, vocabulary.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, decoder1, 75000,plot_every=50,  print_every=50)

#with attn only
if USE_ATTN:
    evaluateRandomly(encoder1, decoder1)


from __future__ import unicode_literals, print_function, division
from get_data import make_list,make_pairs

import rnn
import random
import torch.nn as nn
from torch import optim
import torch
import torchtext.vocab as vocab
import numpy as np
import time
import math
from utils import masked_cross_entropy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO GPU / CPU numpy ?

#Constants
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 4
epochs = 10000
batch_size = 4
USE_ATTN = False
rgb_number_of_dim = 3
teacher_forcing_ratio = 0 #no need
mu = 0.1 # for enforcing RGB distance
clip = 50.0


#Getting data (list of color descriptions)
colors= make_list()
#training pairs, vocabulary, dictionnary with a list of RGB values associated to every color
pairs,vocabulary,RGB = make_pairs(colors,'train')

#set batches

#get weights
s = vocab.GloVe('6B')
sample = s.vectors[0]
# assuming sample is part of vocabulary
embeddings_dim = s.dim #300


#random uniform(Min,Max) initialization of unknown words, else init. with gloVe embeddings
# what about an 'UNKOWN' token ?
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
#embeddings maps indices to vectors in embedding space (300 dimensions)
embeddings = torch.from_numpy(embeddings).float()


def RGB_dist(target_lengths,input_tensors,encoder_outputs) :
        # 3x64x3 print(encoder_output.shape)
        batch_size = len(target_lengths)
        sum_distances = 0
        for i in range(batch_size) :
            input_color_string = ''
            target_length = target_lengths[i]
            input_tensor = [input_tensors[j][i] for j in range(target_length)] #dont count pads !
            encoder_output = encoder_outputs[target_length-1][i] #last rgb value
            #print(encoder_output)
            for i in range(target_length) :
                if input_tensor[i] not in  (EOS_token , PAD_token) :
                    input_color_string += vocabulary.index2word[input_tensor[i].item()] + ' '
            input_color_string = input_color_string[:-1]

            #random RGB value
            idx = random.randrange(len(RGB[input_color_string]))
            target_rgb = np.array(RGB[input_color_string][idx])
            # the RGB values have to be in [-0.5 , 0.5] (-0.5) !
            #TODO ignore pads ??
            #print(encoder_output)
            #print(target_rgb)
            distance =  np.linalg.norm( np.subtract(np.square(encoder_output.data),np.square(target_rgb)),2)
            sum_distances += mu * distance

        return sum_distances / batch_size

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)


    # Prepare input and output variables
    decoder_input = torch.LongTensor([SOS_token] * batch_size).to(device)
    decoder_hidden = encoder_hidden#[:-1]  # Use last (forward) hidden state from encoder #TODO

    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).to(device)

    # Move new Variables to CUDA
    decoder_input.to(device)
    encoder_outputs.to(device)
    #print(target_lengths)
    #print(encoder_outputs) #take last encoder outputs (depends on size)
    # Run through decoder one time step at a time
    last_RGB_values = []
    last_RGB_values_indices = np.array(target_lengths) - 1
    for i in range(len(target_lengths)):
        last_RGB_values.append(encoder_outputs[last_RGB_values_indices[i], i, :])
    last_RGB_values = torch.stack(last_RGB_values).to(device) #our hidden size
    #TODO hidden state is always the last RGB value ?
    last_RGB_values = (last_RGB_values.unsqueeze(0)).to(device)
    #print(last_RGB_values[0])
    for t in range(max_target_length):

        decoder_output, decoder_hidden = decoder(decoder_input, last_RGB_values) #encoder outputs for attn
        #print(decoder_output.size())
        all_decoder_outputs[t] = decoder_output
        topv, topi = decoder_output.topk(1)  # maximum prediction of index
        decoder_input = topi.squeeze().detach()  # detach from history as input
        #decoder_input = target_batches[t]  # Next input is target = teacher forcing (could be prediction)

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    distance = RGB_dist(target_lengths,input_batches,encoder_outputs)

    loss += distance
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
#     #Get hidden state
#     encoder_hidden = encoder.initHidden()
#     #Clears the gradients
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)
#
#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#     loss = 0
#
#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]
#
#     #start of seq
#     decoder_input = torch.tensor([[SOS_token]], device=device)
#
#     decoder_hidden = encoder_hidden
#     #teacher forcing for faster training !
#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#
#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             if(USE_ATTN) :
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#             else :
#                 decoder_output, decoder_hidden = decoder(
#                     decoder_input, decoder_hidden)
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di]  # Teacher forcing
#
#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             #TODO
#             if (USE_ATTN):
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#             else:
#                 decoder_output, decoder_hidden = decoder(
#                     decoder_input, decoder_hidden)
#             #print(encoder_output)
#             topv, topi = decoder_output.topk(1) #maximum prediction of index
#             decoder_input = topi.squeeze().detach()  # detach from history as input
#
#             loss += criterion(decoder_output, target_tensor[di])
#             if decoder_input.item() == EOS_token:
#                 break
#
#
#     #computes all variables with regard to loss
#     loss.backward()
#     # perform updates using calculated gradients
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item() / target_length


## utility functions

def indexesFromDescription(vocabulary, description):
    return [vocabulary.word2index[word] for word in description.split(' ')]

def tensorFromDescription(vocabulary, description):
    indexes = indexesFromDescription(vocabulary, description)
    #PADDING
    #padding =  [PAD_token]  * ( MAX_LENGTH - len(indexes) - 1)
    #indexes = padding + indexes
    #print(indexes)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexesFromDescription(vocabulary, pair[0]))
        target_seqs.append(indexesFromDescription(vocabulary, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = (torch.LongTensor(input_padded)).transpose(0, 1).to(device)
    target_var = (torch.LongTensor(target_padded)).transpose(0, 1).to(device)


    return input_var, input_lengths, target_var, target_lengths


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

    #uniformly taken from pairs -> later random choice from RGB values
    #training_pairs = [tensorsFromPair(random.choice(pairs))
                      #for i in range(n_iters)]
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):

        #training_pair = training_pairs[iter - 1]
        #setting output size = input size => what about padding
        #the same
        input_tensor = input_batches
        target_tensor = target_batches


        loss, _, __ = train(
        input_batches, input_lengths, target_batches, target_lengths,encoder, decoder,encoder_optimizer, decoder_optimizer,criterion)
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





encoder1 = rnn.EncoderRNN(vocabulary.n_words, rgb_number_of_dim,embeddings).to(device)
if(not USE_ATTN) :
    decoder1 = rnn.DecoderRNN(rgb_number_of_dim,embeddings,vocabulary.n_words).to(device)
else :
    decoder1 = rnn.AttnDecoderRNN(rgb_number_of_dim, vocabulary.n_words, embeddings, dropout_p=0.1).to(device)

trainIters(encoder1, decoder1, epochs,plot_every=50,  print_every=50)

#with attn only
if USE_ATTN:
    evaluateRandomly(encoder1, decoder1)


#path_encoder = '/scratch/hamdans/project/enc'
#torch.save(encoder1, path_encoder)

#path_decoder = '/scratch/hamdans/project/dec'
#torch.save(decoder1, path_decoder)


import os
dirpath = os.getcwd()
encoder_path = dirpath + '/enc'
decoder_path = dirpath + '/dec'
torch.save(encoder1, encoder_path)
torch.save(decoder1, decoder_path)


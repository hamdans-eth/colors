from __future__ import unicode_literals, print_function, division
from get_data import *

import rnn4
import random
import torch.nn as nn
from torch import optim
import torch
import torchtext.vocab as vocab
import numpy as np
import time
import math
from utils import get_priors_, CustomTrace,cholesky

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constants
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 4
epochs = 15000
rgb_dim_n = 3
embedding_dim_n = 300
learning_r = 0.05
alpha = 0.05 # for KL divergence
trace = CustomTrace()
grad_clipping = 10

#Getting data (list of color descriptions)
colors= make_list()
#training pairs, vocabulary, dictionnary with a list of RGB values associated to every color
pairs,vocabulary,RGB = make_pairs(colors,'train')

means,log_variances=  get_priors_(RGB)
for k,v in log_variances.items() :
    s = np.all(np.linalg.eigvals(v) > 0)
    if not s : print('not')


#get weights
s = vocab.GloVe('6B')
sample = s.vectors[0]
# assuming sample is part of vocabulary
embeddings_dim = s.dim #300


#random uniform(Min,Max) initialization of unknown words, else init. with gloVe embeddings

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

def tensor_to_string (input_tensor):
    input_color_string = ''
    for i in range(tensor_length(input_tensor)):
        word = vocabulary.index2word[input_tensor[i].item()]
        if word != 'PAD' and word !='EOS':
            input_color_string += word + ' '

    return  input_color_string[:-1]

def tensor_length(input_tensor) :
    #count = 0
    #for element in input_tensor :
    #    count += 1
    #print(input_tensor.shape[0])
    return input_tensor.shape[0]

def RGB_dist(input_tensor,encoder_output) :
        input_color_string = tensor_to_string(input_tensor)

        #random RGB value
        #squared
        idx = random.randrange(len(RGB[input_color_string]))
        target_rgb = np.square(np.array(RGB[input_color_string][idx]))
        target_rgb = torch.Tensor(target_rgb).to(device)**2
        distance = target_rgb.sub(encoder_output**2)
        distance = torch.sum(torch.norm(distance, p=2))
        return distance #  mu * distance

def end_padding(input_tensor) :
    #print(tensor_to_string(input_tensor))
    space = -(tensor_length(input_tensor) - 2) + MAX_LENGTH
    #print(space)
    if (space > 0):
        end_pads =  torch.tensor([[PAD_token]] * space,dtype=torch.long)
        input_tensor = torch.cat((input_tensor,end_pads),0)
    #print(input_tensor)
    return 0


def mm_triangular(v,var) :
    result = torch.zeros(3)
    a = [ v[0].mul(var[0]) + v[1].mul(var[1]) + v[2].mul(var[3]),
          v[1].mul(var[2]) + v[2].mul(var[4]),
          v[2].mul(var[5]) ]
    for i in range(3) :
        result[i] = result[i] + a[i]
    result = result.unsqueeze(0).t()

    return result

def sample_z(mu, var):
    #var vector of length 6 such that
    # a_0   0       0
    # a_1   a_2     0
    # a_3   a_4     a_5
    #reparam trick


    eps = torch.randn((3,1), requires_grad=True)
    result = (mu + mm_triangular(eps,var) ).squeeze()

    return result

def get_distance(input_tensor,current_RGB) :
    input_color_string = ''
    #chose a random RGB value
    for i in range(MAX_LENGTH):
        word = vocabulary.index2word[input_tensor[i].item()]
        if word != 'PAD' and word != 'EOS':
            input_color_string += word + ' '
    input_color_string = input_color_string[:-1]

    idx = random.randrange(len(RGB[input_color_string]))
    target_rgb = np.array(RGB[input_color_string][idx])
    #TODO change the distance
    current_RGB = current_RGB.detach().numpy()[0]
    #print(current_RGB)
    distance = mu * np.linalg.norm(np.subtract(current_RGB, target_rgb), 2)
    return mu * distance

def det(a):
    a = a.clone()
    idx = [0,2,5]
    det = torch.Tensor([1.])
    for i in idx :
        det = det * a[i]
    return det.pow(2)

def sq_triangular(a):
    a = a.clone()
    array =[[a[0].pow(2) , a[0].mul(a[1]) , a[0].mul(a[3])],
                          [a[0].mul(a[1]) , a[2].pow(2) + a[1].pow(2), a[2].mul(a[4])],
                          [a[0].mul(a[3]), a[2].mul(a[4]),a[3].pow(2) + a[4].pow(2) + a[5].pow(2)]]

    squared = torch.zeros([3,3])
    for i in range(3):
        for j in range(3) : squared[i][j] = squared[i][j] + array[i][j]

    return squared



def kl_loss(mu_z,C_z,mu_t,cov_t) :
    #KL(p(z|x,y) || N(mu_y, cov_y))
    cov_t = torch.tensor(cov_t,requires_grad=True).float()
    cov_z =sq_triangular(C_z)
    det_z = det(C_z)
    mu_t = torch.Tensor(mu_t).view(-1,1)
    det_t = torch.potrf(cov_t).diag().prod()
    cov_t = torch.Tensor(cov_t)
    cov_t_inv = cov_t.inverse()
    diff = mu_t - mu_z
    square = torch.mm(diff.t(),cov_t_inv)
    square = torch.mm(square,diff)
    result = 0.5 * ( torch.log(det_t/det_z) - rgb_dim_n + trace(torch.mm(cov_t_inv,cov_z)) + square)
    return alpha * result



def train(input_tensor, target_tensor, encoder, decoder,linear, encoder_optimizer, decoder_optimizer, linear_optimizer,
          criterion, max_length=MAX_LENGTH):

    #Get hidden state
    encoder_hidden = encoder.initHidden()
    #Clears the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    linear_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = tensor_length(input_tensor)
    loss = 0

    #First we iterate through the words feeding tokens & last hidden state
    #print(input_length)
    for i in range(input_length):
        mu_z,sigma_z, encoder_hidden = encoder(input_tensor[i], encoder_hidden)


    #Get the last RGB value
    #print(torch.nn.functional.sigmoid(sample_z(mu_z,log_sigma_v)))
    target = tensor_to_string(input_tensor)
    mu_t = means[target]
    log_sigma_t = log_variances[target]
    #print(mu_z,log_sigma_z)
    #print(mu_t,log_sigma_t)


    current_RGB = sample_z(mu_z, sigma_z) # can be out of unit interval !!


    RGB_hidden = linear(current_RGB)
    decoder_hidden = RGB_hidden.clone()


    #start of seq
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # fill the vector of predicted output
    prediction = []
    #end_padding(input_tensor)
    #print(target_length)
    for di in range(target_length):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di])
        prediction = [prediction + [topi]]
        decoder_hidden = decoder_hidden + RGB_hidden # to get more info from RGB


    #LOSS
    kl = kl_loss(mu_z,sigma_z,mu_t,log_sigma_t)
    rec = loss / (target_length - 1)
    loss = rec  + kl # minus sign ?
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clipping)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clipping)
    torch.nn.utils.clip_grad_norm(linear.parameters(), grad_clipping)

    encoder_optimizer.step()
    decoder_optimizer.step()
    linear_optimizer.step()
    #print(loss.item()) #/ float(target_length) )
    return loss.item() , kl.detach().numpy(),rec.detach().numpy() #/ float(target_length)  #+ distance


## utility functions

def indexesFromDescription(vocabulary, description):
    return [vocabulary.word2index[word] for word in description.split(' ')]

def tensorFromDescription(vocabulary, description):
    indexes = [PAD_token] + indexesFromDescription(vocabulary, description)
    # [PAD_token for _ in range(empty_spaces)] + indexes
    #print(indexes)
    #empty_spaces = MAX_LENGTH - len(indexes) #the EOS
    #if empty_spaces > 0 : indexes = [PAD_token] + indexes #[PAD_token for _ in range(empty_spaces)] + indexes
    # one pad ?
    indexes.append(EOS_token)

    #print(indexes)
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



def trainIters(encoder, decoder,linear, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print_kl_total = 0
    print_rec_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    linear_optimizer = optim.SGD(linear.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]


    criterion = nn.NLLLoss(ignore_index=2)

    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]
        #setting output size = input size => what about padding
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]


        loss,kl,rec = train(input_tensor, target_tensor, encoder,
                     decoder, linear, encoder_optimizer, decoder_optimizer, linear_optimizer, criterion)

        print_kl_total += kl
        print_rec_total += rec

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            print_rec_total = print_rec_total / print_every
            print_kl_total = print_kl_total / print_every
            print('average kl = ' + str(print_kl_total))
            print('average reconstruction = ' + str(print_rec_total))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            print_kl_total = 0
            print_rec_total = 0
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




encoder = rnn4.EncoderRNN(vocabulary.n_words, rgb_dim_n,embeddings).to(device)
decoder = rnn4.DecoderRNN(embedding_dim_n,embeddings,vocabulary.n_words).to(device)
linear = rnn4.RGB_to_Hidden(rgb_dim_n, embedding_dim_n).to(device)

trainIters(encoder, decoder,linear, epochs,plot_every=500,  print_every=500,learning_rate=learning_r)


import os
dirpath = os.getcwd()
encoder_path = dirpath + '/enc_cov'
decoder_path = dirpath + '/dec_cov'
torch.save(encoder, encoder_path)
torch.save(decoder, decoder_path)
lin_path = dirpath + '/lin_cov'
torch.save(linear,lin_path)


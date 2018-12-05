from __future__ import unicode_literals, print_function, division
from get_data import *

import model
import random
import torch.nn as nn
from torch import optim
from utils import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constants
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 4
epochs = 50000
latent_space = 3
embedding_space = 300
learning_r = 0.00005
alpha = 0.5 # for penalizing KL divergence
trace = CustomTrace()
grad_clipping = 10
SAVE = True

#Getting data (list of color descriptions)
colors = make_list()
#training pairs, vocabulary, dictionnary with a list of RGB values associated to every color
pairs,vocabulary,RGB = make_pairs(colors,'tarin')
#statistical data on RGB values
means,variances =  get_priors_(RGB)

#get weights
#random uniform(Min,Max) initialization of unknown words, else init. with gloVe embeddings
embeddings = get_embeddings(vocabulary,embedding_space)

def kl_loss(mu_z,C_z,mu_t,cov_t) :
    #KL(p(z|x,y) || N(mu_y, cov_y))
    cov_t = torch.tensor(cov_t).float()
    cov_z = torch.mm(C_z,C_z.t())
    det_z = torch.diag(C_z).prod() ** 2
    det_t = torch.potrf(cov_t).diag().prod() ** 2
    mu_t = torch.Tensor(mu_t).view(-1,1)
    cov_t_inv = cov_t.inverse()
    diff = mu_t - mu_z
    square = torch.mm(diff.t(),cov_t_inv)
    square = torch.mm(square,diff)
    result = 0.5 * (torch.log(det_t/det_z) - latent_space + trace(torch.mm(cov_t_inv,cov_z)) + square)
    return alpha * result

def train(input_tensor, target_tensor, encoder, decoder,linear, encoder_optimizer, decoder_optimizer, linear_optimizer,
          criterion, max_length=MAX_LENGTH):

    #Get hidden state initialized
    encoder_hidden = encoder.initHidden()

    #Clears the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    linear_optimizer.zero_grad()

    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]
    rec_loss = 0
    # First we iterate through the words feeding tokens & last hidden state at each time step
    # mean and square of the latent covariance outputted
    for i in range(input_length):
        mu_z,C_z, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

    #Get the last RGB value sampled from latent distribution
    target = tensor_to_string(input_tensor,vocabulary)
    sample = sample_z(mu_z, C_z) # can be out of unit interval !!

    #Initialize the hidden state of the decoder with a linear transformation of the encoded latent space
    transformed_latent = linear(sample)
    decoder_hidden = transformed_latent.clone()

    #start of sequence pad
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # fill the vector of predicted output
    prediction = []

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        #compute the softmax loss with outputted token
        rec_loss += criterion(decoder_output, target_tensor[di])

        #building the output string
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        prediction = [prediction + [topi]]

        #the hidden state gets the latent space
        decoder_hidden = decoder_hidden + transformed_latent


    #Loss = alpha * divergence + reconstruction error
    mu_t = means[target]
    sigma_t = variances[target]
    kl = kl_loss(mu_z,C_z,mu_t,sigma_t)
    rec_loss = rec_loss / (target_length - 1)
    loss = kl + rec_loss

    #udpate step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clipping)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clipping)
    torch.nn.utils.clip_grad_norm_(linear.parameters(), grad_clipping)

    encoder_optimizer.step()
    decoder_optimizer.step()
    linear_optimizer.step()
    return loss.item() , kl[0][0].detach().numpy(),rec_loss.detach().numpy()

def trainIters(encoder, decoder,linear, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print_kl_total = 0
    print_rec_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=1000 * learning_rate, momentum=0.5, nesterov=True)
    linear_optimizer = optim.SGD(linear.parameters(), lr=1000 * learning_rate, momentum=0.5, nesterov=True)

    training_pairs = [tensorsFromPair(random.choice(pairs),vocabulary)
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
            print('average kl =  %.4f' % print_kl_total)
            print('average reconstruction =  %.4f' % print_rec_total)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            print_kl_total = 0
            print_rec_total = 0


encoder = model.EncoderRNN(vocabulary.n_words, latent_space,embeddings).to(device)
decoder = model.DecoderRNN(embedding_space,embeddings,vocabulary.n_words).to(device)
linear = model.RGB_to_Hidden(latent_space, embedding_space).to(device)

trainIters(encoder, decoder,linear, epochs,plot_every=500,  print_every=500,learning_rate=learning_r)

if SAVE :
    dirpath = os.getcwd()
    encoder_path = dirpath + '/enc'
    decoder_path = dirpath + '/dec'
    torch.save(encoder, encoder_path)
    torch.save(decoder, decoder_path)
    lin_path = dirpath + '/lin'
    torch.save(linear,lin_path)


import random
import get_data
from get_data import load_models
from utils import *

PAD_token = 2
EOS_token = 1
SOS_token = 0

COEF = 0.5
N_iters = 1000
N_samples = 10
enc,lin,dec = load_models()
pairs, vocabulary, RGB = get_data.make_pairs(get_data.make_list(),'train')
means,var_t=  get_priors_(RGB)

count = 0
correct = 0
for i in range(N_iters):
    with torch.no_grad():
        # ENCODING PART
        current = []
        rand_idx = random.randrange(len(pairs))
        description = pairs[rand_idx][0]
        mu = torch.Tensor(means[description]).unsqueeze(1)
        var_ = torch.Tensor(var_t[description])
        for _ in range(N_samples) :
            mu_t = torch.mm(var_,torch.randn((3,1)))  + mu #sample_cov(mu, var_)
            encoder_output = mu_t.t()
            RGB_hidden = lin(encoder_output)
            decoder_hidden = RGB_hidden.clone()
            decoder_input = torch.LongTensor([[SOS_token]])

            string = ''
            for t in range(10):
                decoder_output, decoder_hidden = dec(decoder_input, decoder_hidden)  # encoder outputs for attn
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_hidden += RGB_hidden
                word = vocabulary.index2word[topi[0].item()]

                if topi == EOS_token: break
                if t != 0: string += vocabulary.index2word[topi[0].item()] + ' '

            for i, word in enumerate(string.split()):
                if i < len(description.split(' ')) and word == description.split(' ')[i]:
                    correct += 1
                count += 1

            print('decoded description : "' + string + '"')
        print('real decription : "' + description + '"')
        print()
print("%.2f" % (correct/count* 100) +'% accuracy ')


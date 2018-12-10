import random
import get_data
from get_data import load_models
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number_samples", nargs='?',help="number of descriptions generated"
                                                 ,default=1,type=int)
parser.add_argument("-c","--coef",nargs='?',help="coefficient for sampling :"
                                                 "set to zero for zero noise",default=0.1,type=float)
args = parser.parse_args()



PAD_token = 2
EOS_token = 1
SOS_token = 0
COEF = args.coef
N_iters = 10000

if args.coef > 0 :
    N_samples = args.number_samples
else :
    N_samples = 1

enc,lin,dec = load_models()
pairs, vocabulary, RGB = get_data.make_pairs(get_data.make_list(),'test')
means,var_t=get_priors_(RGB)


print('Number of samples :' + str(args.number_samples))
print('Noise coefficient:' + str(args.coef))
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
            check = True
            mu_t = COEF * torch.mm(var_,torch.randn((3,1)))  + mu #sample_cov(mu, var_)
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
                string += vocabulary.index2word[topi[0].item()] + ' '

            for i, word in enumerate(description.split()):
                if i > len(string.split(' ')) - 1 :
                    check = False
                elif word != string.split(' ')[i]:
                    check = False
            correct += int(check)
            count += 1

            print('decoded description : "' + string + '"')
        print('real decription : "' + description + '"')
        print()

print("%.2f" % ((100. * correct)/count) +'% accuracy ')


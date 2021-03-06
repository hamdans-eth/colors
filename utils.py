import torch
import numpy as np
import math
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 4
SOS_token = 0
EOS_token = 1
PAD_token = 2

#m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
def tensorFromDescription(vocabulary, description):
    indices = [vocabulary.word2index[word] for word in description.split(' ')]
    indices.append(EOS_token)
    diff = MAX_LENGTH + 1 - len(indices)
    if diff > 0 : indices.extend([PAD_token for _ in range(diff)])
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

def sample_z(mu,sigma,coef=0.5):
    #reparam trick multivariate
    #coef can be tuned down for less noisy samples
    #random_vector = torch.tensor([m.sample() for _ in range(3)])
    #return (mu + coef*torch.mm(sigma,random_vector) ).squeeze()
    return (mu + coef*torch.mm(sigma,torch.randn((3,1))) ).squeeze()

def tensorsFromPair(pair,vocabulary):
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

class CustomTrace(torch.autograd.Function):
    def forward(self, input):
        self.isize = input.size()
        return input.new([torch.trace(input)])

    def backward(self, grad_output):
        isize = self.isize
        grad_input = grad_output.new(isize).copy_(torch.eye(*isize))
        grad_input.mul_(grad_output[0])
        return grad_input

def tensor_to_string (input_tensor,vocabulary):
    input_color_string = ''
    for i in range(input_tensor.shape[0]):
        word = vocabulary.index2word[input_tensor[i].item()]
        if word != 'PAD' and word !='EOS':
            input_color_string += word + ' '
    return  input_color_string[:-1]

def get_priors_(RGB):
    #return clusters with mu and sigma for each RGB
    means = {}
    variances = {}
    for key in RGB:
        R = [v[0] for v in RGB[key]]
        G = [v[1] for v in RGB[key]]
        B = [v[2] for v in RGB[key]]
        means[key] = np.array([np.mean(R),np.mean(G), np.mean(B)])
        variances[key] = np.cov( np.array([R, G, B]))
    return means,variances

def sample_RGB(mu, var,coef):
    #eps = torch.tensor([[m.sample()] for _ in range(3)])
    eps = torch.randn((3,1))
    sample = (mu.t() + coef * torch.mm(var,eps)) #* mm_triangular(eps,var))
    while (any(sample[0] < 0) or any(sample[0] > 1)):
        #eps = torch.tensor([[m.sample()] for _ in range(3)])
        eps = torch.randn((3, 1))
        sample = (mu.t() +  coef *  torch.mm(var,eps)) #mm_triangular(eps, var))
    return sample

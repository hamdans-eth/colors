from get_data import *
import os
from skimage import io
from matplotlib import pyplot as plt
import random
from utils import *

dirpath = os.getcwd()


DIAGONAL = False #True
RANDOM = False

abs_path = os.path.dirname(os.path.abspath(__file__)) +'/'
path = abs_path +  'dec'
if DIAGONAL : path += '_s'
else : path += '_cov'
decoder = torch.load(path,map_location='cpu')
path = abs_path +   'enc'
if DIAGONAL : path += '_s'
else : path += '_cov'
encoder = torch.load(path,map_location='cpu')
path = abs_path +   'lin'
if DIAGONAL : path += '_s'
else : path += '_cov'
linear = torch.load(path,map_location='cpu')

EOS_token = 1
PAD_token = 2
COEF = 0.5 # how random is sampling
pairs,vocabulary,RGB = make_pairs(make_list(),'test')
means,covs=  get_priors_(RGB)
device = 'cpu'



def sample_cov(mu, var):
    eps = torch.randn((3,1), requires_grad=True)
    sample = (mu.t() + COEF * torch.mm(var,eps))
    while (any(sample[0] < 0) or any(sample[0] > 1)):
        eps = torch.randn((3, 1), requires_grad=True)
        sample = (mu.t() +  COEF *  torch.mm(var,eps))
    return sample

N= 5
colors = ['blue','blue','amber','mushroom','poop']
modifiers = ['very dark ','dark ', '' , 'light ', 'very light ']
x = make_list()
x2 = [description for description in x if len(description.split(' ')) == 2]
x3 = [description for description in x if len(description.split(' ')) == 3]
n = vocabulary.n_words
if RANDOM :
    colors = [random.choice(x).split(' ')[-1] for _ in range(N)]
    #modifiers = [''] +  [random.choice(x2).split(' ')[0] + ' '+ random.choice(x3).split(' ')[1] + ' '  for _ in range(N-1)]
    #modifiers = [''] + modifiers
    modifiers = ['']+ [random.choice(x).split(' ')[0] + ' '  for _ in range(N-1)]


print(colors)
print(modifiers)



all = []
for color in colors :
    current = []
    for modifier in modifiers :
        RGB = []
        input_tensor = tensorFromDescription(vocabulary, modifier + color)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        with torch.no_grad():
            for i in range(input_length):
                mu_z, C_z, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            # Get the last RGB value
            current_RGB = (sample_cov(mu_z,C_z)).detach().numpy()
            current_RGB = current_RGB[0]
            current += [current_RGB]
    all += [current]


all = np.array(all).squeeze()
io.imshow(all)

my_xticks = modifiers
plt.xticks(range(N), my_xticks)
my_yticks = colors
plt.yticks(range(N), my_yticks)
plt.show()

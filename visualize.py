from get_data import *
import os
from skimage import io
from matplotlib import pyplot as plt
import random
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r","--random" ,help="random colors" ,action="store_true")
parser.add_argument("-c","--coef",nargs='?',help="coefficient for sampling :"
                                                 "set to zero for zero noise",default=0.1,type=float)

args = parser.parse_args()
print('Random colors:' + str(args.random))
print('Noise coefficient:' + str(args.coef))


dirpath = os.getcwd()
EOS_token = 1
PAD_token = 2
COEF = args.coef  # how random is sampling
pairs,vocabulary,RGB = make_pairs(make_list(),'test')
means,covs=  get_priors_(RGB)
device = 'cpu'
RANDOM = args.random
encoder,linear,decoder = load_models()


N= 5
colors = ['blue','vermillion','amber','mushroom','poop']
modifiers = ['very dark ','dark ', '' , 'light ', 'very light ']
x = make_list()
x1 = [description for description in x if len(description.split(' ')) == 1]
x2 = [description for description in x if len(description.split(' ')) == 2]
x3 = [description for description in x if len(description.split(' ')) == 3]
n = vocabulary.n_words
if RANDOM :
    colors = [random.choice(x1).split(' ')[0] for _ in range(N)]
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
            current_RGB = (sample_RGB(mu_z,C_z,COEF)).detach().numpy()
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

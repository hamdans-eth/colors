# Grounded language learning of visual-lexical color descriptions
The goal is to learn from description-color pairs to describe colors, and to create colors from descriptions. A gaussian mixture is learnt in the latent RGB space instead of the centered normal gaussian of the VAE (Kingma et al. 2013). 
<img src="https://image.noelshack.com/fichiers/2019/15/1/1554717389-capture-d-ecran-2019-04-08-a-11-55-52.png" alt="The encoder part" width="500" class="center"/>


## What's needed
 - python 3.6
 - pytorch 0.4
 - torchtext 0.4
 - skimage 0.14
 - matplotlib 3.0
 - _GPU not needed_ (short sequences of word and low dimension latent space seem to not benefit from GPU parallelization)

## Training a model
Run `main.py --save` will run a model with parameters and save it.


## Testing : visualization
Run `visualize.py` (loads what has been saved as model).
This will display a grid of modifiers (y-axis) and colors (x-axis).

 - you can tune the noise coefficient of the sampling with `-c` argument (default: `0.1`)
 - you can get random mix of colors and modifiers with `-r` (default: `False`)

<img src="https://image.noelshack.com/fichiers/2019/15/1/1554717389-capture-d-ecran-2019-04-08-a-11-55-04.png" alt="Decoding" width="500"/>
Example of colors generated from random mix of adjectives and colors

## Testing : generating color descriptions
Run `test.py` (loads what has been saved as model).
Select a set of RGB values and outputs a description and then compares it with the corresponding description of the test set.
The accuracy is displayed (exact match).

 - you can tune the noise coefficient of the sampling with `-c` argument (default: `0.1)`
 - you can chose the number of sequences generated for each RGB value with `-n` (default :`1`)
 
 <img src="https://image.noelshack.com/fichiers/2019/15/1/1554717389-capture-d-ecran-2019-04-08-a-11-55-34.png" alt="Decoding" width="500"/>
Comparison between deterministic and stochastic auto-encoder description predictions randomly sampled from the orange red color distribution in the RGB space.


 

## Other files

 - `get_data.py` provides function to load the needed data
 - `utils.py` furnishes helpful functions 
 - `model.py` describes the modified autoencoder architecture

## Source
the dataset used and the utility functions used to load it are from http://mcmahan.io/lux/

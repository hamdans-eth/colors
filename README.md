# Grounded language learning of visual-lexical color descriptions
The goal is to learn from description-color pairs to describe colors, and to create colors from descriptions. A gaussian mixture is learnt in the latent RGB space instead of the centered normal gaussian of the VAE (Kingma et al. 2013). 

<img src="https://image.noelshack.com/fichiers/2019/15/1/1554715859-12331.png" alt="The encoder part"/>


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

<img src="https://image.noelshack.com/fichiers/2019/15/1/1554717154-capture-d-ecran-2019-04-08-a-11-50-55.png" alt="Decoding"/>
Example of colors generated from random mix of adjectives and colors
## Testing : generating color descriptions

Run `test.py` (loads what has been saved as model).
Select a set of RGB values and outputs a description and then compares it with the corresponding description of the test set.
The accuracy is displayed (exact match).

 - you can tune the noise coefficient of the sampling with `-c` argument (default: `0.1)`
 - you can chose the number of sequences generated for each RGB value with `-n` (default :`1`)
 
 <img src="https://image.noelshack.com/fichiers/2019/15/1/1554716761-capture-d-ecran-2019-04-08-a-11-40-47.png" alt="Decoding"/>
Example from description predictions randomly sampled from the color _orange red_ distribution in the RGB space.


 

## Other files

 - `get_data.py` provides function to load the needed data
 - `utils.py` furnishes helpful functions 
 - `model.py` describes the modified autoencoder architecture

## Source
the dataset used and the utility functions used to load it are from http://mcmahan.io/lux/

# Grounded language learning of visual-lexical color descriptions
The goal is to learn from description-color pairs to describe colors, and to create colors from description. 

## What's needed
 - python 3.6
 - pytorch 0.4
 - torchtext 0.4
 - skimage 0.14
 - matplotlib 3.0
 - _GPU not needed_

## Training a model
Run `python main.py --save` will run a model with parameters and save it.


## Testing : visualization
Run `visualize.py` (loads what has been saved as model).
This will display a grid of modifiers (y-axis) and colors (x-axis).

 - you can tune the noise coefficient of the sampling with `-c` argument (default: `0.1`)
 - you can get random mix of colors and modifiers with `-r` (default: `False`)

## Testing : generating color descriptions

Run `test.py` (loads what has been saved as model).
Select a set of RGB values and outputs a description and then compares it with the corresponding description of the test set.
The accuracy is displayed (exact match).

 - you can tune the noise coefficient of the sampling with `-c` argument (default: `0.1)`
 - you can chose the number of sequences generated for each RGB value with `-n` (default :`1`)

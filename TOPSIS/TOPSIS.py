'''
Docstring for TOPSIS.py

It's a python file running TOPSIS algorithm based on package pyDecision.
Further information about TOPSIS_method can be found in TOPSIS.ipynb file under the same folder.

'''

import numpy as np

import topsis_modified as alg

# TOPSIS

# Weights
weights =  [0.25, 0.25, 0.25, 0.25]

# Load Criterion Type: 'max' or 'min' or 'interval' or 'nominal'
criterion_type = ['max', ['interval', 5, 3], ['nominal', 6], 'max']

# Dataset
dataset = np.array([
                [6, 8, 4, 7],   #a1
                [9, 3, 4, 6],   #a2
                [4, 9, 7, 3],   #a3
                [8, 2, 5, 8],   #a4
                [4, 9, 2, 3],   #a5
                [7, 5, 9, 9],   #a6
                [9, 6, 3, 1],   #a7
                [3, 5, 7, 6],   #a8
                [5, 3, 8, 5],   #a9
                [4, 6, 3, 8],   #a10
                ])

# Call TOPSIS. The official topsis_method isn't robust enough, 
# so we use the modified_topsis, able to handle non-polarized data
relative_closeness = alg.topsis_method(dataset, weights, criterion_type,   #literal meaning
                                        plot_bar=True, plot_geom=True,     #to draw pr not
                                        alt_labels=None,                   #labels for alternatives
                                        verbose=True,                      #verbose or not (extra contents like warnings and alternatives ci value)
                                        save_prefix='test',                #prefix.name for saving plots. suffix.name seen at topsis_modified.py
                                        show_plots=True)                   #to plt.show() or not
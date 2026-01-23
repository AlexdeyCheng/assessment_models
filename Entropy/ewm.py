# Required Libraries
import numpy as np

import entropy_modified as em

# Load Criterion Type: 'max' or 'min' or 'interval' or 'nominal'
criterion_type = ['max', ['interval', 5, 3], ['nominal', 4], 'max']

# Dataset
dataset = np.array([
                # g1   g2   g3   g4
                [250,  16,  12,  5],
                [200,  16,  8 ,  3],
                [300,  32,  16,  4],
                [275,  32,  8 ,  4],
                [225,  16,  16,  2]
                ])

# Call Entropy Function
weights = em.entropy_method(dataset, criterion_type)

# Weigths
for i in range(0, weights.shape[0]):
  print('w(g'+str(i+1)+'): ', round(weights[i], 3))
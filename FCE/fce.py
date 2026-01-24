import numpy as np
import matplotlib.pyplot as plt
import fce_package as pkg

weights =  [0.10, 0.20, 0.30, 0.40]

criterion_names = ['weight', 'length', 'height', 'age'] #未生成的在子功能性函数中自动补全

criterion_type = ['max', 'max', 'max', 'max']

grade_names = ['g1', 'g2', 'g3', 'g4']  #criiteria grades

sample_names = ['george', 'marx', 'anna']

theme_set = 'green_purple'   
'''
themes = 
        'blue_grey'
            'cmap': 'Blues',
            # Deep Blue, Blue Grey, Light Blue, Steel Blue, Slate Grey
            'colors': ['#2E5B88', '#7A8B99', '#B0C4DE', '#4682B4', '#708090']
        
        'green_purple' 
            'cmap': 'viridis', 
            # Purple, Teal, Green, Light Green, Dark Purple
            'colors': ['#482677', '#2D708E', '#20A387', '#73D055', '#440154']
        
        'grey_yellow'
            'cmap': 'cividis', 
            # Dark Grey, Gold, Grey, Dark Golden Rod, Black
            'colors': ['#404040', '#D4AF37', '#808080', '#B8860B', '#000000']
'''

# Dataset
#Rows of dataset are samples, columns are criteria
dataset = np.array([
#               c1 c2 c3 c4
                [9, 3, 4, 7],   #sample1

                [6, 3, 4, 6],   #s2 

                [4, 9, 7, 3],   #s3

                ])


print(
    pkg.fuzzy_comprehensive_method(criterion_names, criterion_type, grade_names, dataset, weights, sample_names,

                                    theme = theme_set, save_prefix=None, 
                                    graph_heat=False, graph_radar=False
                                    )        
)


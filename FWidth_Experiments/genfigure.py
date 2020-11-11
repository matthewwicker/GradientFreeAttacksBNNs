import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()
import matplotlib.font_manager
from matplotlib.pyplot import figure
figure(num=None, figsize=(5, 5), dpi=100)
sns.set_context("paper", rc={'axes.labelsize': 17.6,
 'axes.titlesize': 19.2,
 'font.size': 19.2,
 'grid.linewidth': 1.6,
 'legend.fontsize': 16.0,
 'lines.linewidth': 2.8,
 'lines.markeredgewidth': 0.0,
 'lines.markersize': 11.2,
 'patch.linewidth': 0.48,
 'xtick.labelsize': 16.0,
 'xtick.major.pad': 11.2,
 'xtick.major.width': 1.6,
 'xtick.minor.width': 0.8,
 'ytick.labelsize': 16.0,
 'ytick.major.pad': 11.2,
 'ytick.major.width': 1.6,
 'ytick.minor.width': 0.8}) 

sns.set_style('darkgrid')
font = {'weight' : 'bold', "size":12}
matplotlib.rc('font', **font)

import numpy as np

# Names of the Bars 
names = [128, 256, 512, 1024]

# IBP-ROBUST TRAINING FIGURES
accuracy = [0.9480000138282776, 0.9639999866485596, 0.9599999785423279, 0.9760000109672546]
upper_bounds = [0.4000000059604645, 0.6779999732971191, 0.7379999756813049, 0.7720000147819519]


plt.figure(figsize=(8,5), dpi=200)
sns.barplot(names, upper_bounds)
for i in range(len(accuracy)):
    plt.plot(i, accuracy[i], marker='*')
plt.ylim([0.0, 1.0])
plt.title(r'CW Robustness (BBB / $\epsilon$ = 0.1)')
plt.ylabel('Robustness')
plt.xlabel('Width of NN')

plt.savefig('WidthFigExample.png')

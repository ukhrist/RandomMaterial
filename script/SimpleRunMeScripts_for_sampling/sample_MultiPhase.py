from math import pi, sqrt
from turtle import color
import matplotlib
from tqdm import tqdm
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.DataManager import read_data3D
from source.MultiPhaseFields.MultiPhaseMaterial import MultiGrainMaterial
from source.Kernels.MaternKernel import MaternKernel


###=============================================
### Configuration parameters

config = {
    'ndim'              :   2,
    'grid_level'        :   9,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   10,
    'correlation_length':   0.01,
    'Folded_GRF'        :   False,
### Support
    'alpha'             :   0.6,
### Grains
    'nPhases'           :   10,
    'SizeCell'          :   [52,64],
    'angle'             :   pi/3,
    'mask'              :   False,
### Clusters
    'clusters'          :   {
        'apply'                 :   True,
        'ClusterRadius'         :   0.01,
        'scale'                 :   1000.,
        'nu'                    :   10,
        'correlation_length'    :   0.005,
        'Folded'                :   False,
        'ClusterFunctionType'   :   'Gauss'
    },        
}

### Sampling
nsamples = 1
EXPORTDIR = "./"


###=============================================
### Main part

RM = MultiGrainMaterial(**config)
# RM.seed(0) ### fix the random seed in order to have always the same realizations 

import numpy as np
np.random.seed(0)

### Generate multiple samples
for isample in tqdm(range(nsamples)):
    filename = os.path.abspath(os.path.join(EXPORTDIR, f"sample_{isample+1}.png"))
    RM.save_png(filename)

### Generate labeled samples
print()
for isample in range(nsamples):
    Sample, Label = RM.sample_labeled()
    print(f"{isample} sample label: ")
    print(Label)
print()


### Generate one nice looking sample
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['image.cmap']='rainbow'
Sample = RM.sample_numpy()
norm = plt.Normalize(Sample.min(),Sample.max())
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",[
    (0, "mediumvioletred"),
    (1/9, "blueviolet"),
    (2/9, "blue"),
    (3/9, "lightsteelblue"),
    (4/9, "paleturquoise"),
    (5/9, "lawngreen"),
    (6/9, "yellow"),
    (7/9, "royalblue"),
    (8/9, "coral"),
    (1, "crimson"),
    # "blueviolet",
    # "blue",
    # "royalblue",
    # "lightsteelblue",
    # "paleturquoise",
    # "lawngreen",
    # "yellow",
    # "orange",
    # "coral",
    # "crimson",
    # "mediumvioletred",
    ])
plt.imshow(Sample, cmap=cmap, norm=norm)
plt.contour(Sample, config['nPhases'], colors="black", linewidths=0.3)
plt.axis('off')
filename  = os.path.abspath(os.path.join(EXPORTDIR, "sample_MF.jpg"))
plt.savefig(filename, bbox_inches='tight')



print(f"Samples are successfully saved in {os.path.abspath(EXPORTDIR)}")



###=============================================
plt.show()


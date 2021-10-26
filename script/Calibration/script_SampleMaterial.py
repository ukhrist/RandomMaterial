
from source.RMNet.Calibration.DistanceMeasure import DistanceMeasure
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress
import os

from source.RMNet.GaussianRandomField import GaussianRandomField
from source.RMNet.TwoPhaseMaterial import GaussianMaterial
from source.RMNet.RandomMaterial import RandomMaterial
from source.RMNet.StructureFields import GrainStructure, LatticeStructure, MatrixStructure
from source.RMNet.StructureFields.SupportedMaterials import LatticeMaterial, GrainMaterial, ParticlesMaterial
from source.RMNet.Kernels.MaternKernel import MaternKernel
from source.RMNet.Calibration.Calibration_LBFGS import calibrate
from source.RMNet.DataManager import read_data3D
from source.RMNet.StatisticalDescriptors.interface import interface
from source.RMNet.StatisticalDescriptors.curvature import num_curvature



config = {
    'verbose'           :   True,
    'debug'             :   True,
### General
    'ndim'              :   3,
    'grid_level'        :   8,
    'window_margin'     :   0,
    'vf'                :   0.2,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   10,
    'correlation_length':   0.03, #0.0067, #0.025,
    'Folded_GRF'        :   False,
### Mutlti-phase
    'nPhases'           :   3,
    'alpha'             :   0*0.04, #0.0315,
    'tau'               :   0,
### Particles
    'radius'            :   0.25,
    'Poisson_mean'      :   0,
### Lattice
    'nCells'            :   1,
    'SizeVoid'          :   0,
    'thickness'         :   0.16,
    'noise_sparsity'    :   3,
### Data
    'source_parameters' :   'config', ### 'file', 'config'
    'input_folder'      :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
    'output_folder'     :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
}
test_seed = None

MaterialClass = LatticeMaterial
# MaterialClass = GaussianMaterial
# MaterialClass = ParticlesMaterial

########################################################

RM = MaterialClass(**config)
RM.seed(test_seed)
RM.calibration_regime = False

if config['source_parameters'] != 'config':
    if config['source_parameters'] == 'file':
        filename = 'inferred_parameters'
    else:
        filename = config['source_parameters']
    filename = config['input_folder'] + filename

    print('Parameters are imported from ' + filename)
    RM.import_parameters(filename)


X  = RM.sample()

if RM.ndim == 3:
    I  = interface(X)
    H, K, _ = num_curvature(X)
    RM.save_vtk(config['output_folder']+'sample_test', Sample={'phase':X, 'interface':I, 'H':H, 'K':K})
    print('Samples saved to ', os.path.abspath(config['output_folder']))

    measure = DistanceMeasure(**config)
    measure.print(measure.statistial_descriptors(X))

elif RM.ndim == 2:
    fig, axs = plt.subplots(1,2)
    axs[0].set_title('target')
    axs[1].set_title('result')
    RM.plot(ax=axs[1], show=True, Sample=Sample.detach().numpy())

########################################################

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
from source.RMNet.StatisticalDescriptors.common import interpolate

from script.Calibration.load_data import load_data_from_npy

torch.autograd.set_detect_anomaly(True)


config = {
    'verbose'           :   True,
    'debug'             :   True,
### General
    'ndim'              :   3,
    'grid_level'        :   7,
    'window_margin'     :   0,
    'vf'                :   0.2,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   2,
    'correlation_length':   0.1, #0.025,
    'Folded_GRF'        :   False,
### Mutlti-phase
    'nPhases'           :   3,
    'tau'               :   0.01,
### Particles
    'radius'            :   0.05,
    'Poisson_mean'      :   20,
### Lattice
    'nCells'            :   1,
    'SizeVoid'          :   0,
    'alpha'             :   0.1,
    'thickness'         :   0.3,
    'noise_sparsity'    :   0,
### Optimization
    'max_iter'          :   100,
    'GAN'               :   False,
    'SGD'               :   False,
    'mean_only'         :   True,
    'beta'              :   0.9,
    'history_size'      :   100,
    'curvature_eps'     :   0.01,
    'init_batch_size'   :   4,
    # 'nepochs'           :   20,
    # 'Optimizer'         :   torch.optim.LBFGS, ### torch.optim.LBFGS, torch.optim.SGD
    'lr'                :   1,
    # 'adapt'             :   True,
    'tol'               :   1.e-4,
    'line_search'       :   'Armijo', ### 'Armijo', 'None', 'Wolfe'
    'Powell_dumping'    :   False,
    'regularization'    :   1.e-3,
    'not_require_grad'  :   [   
                                # 'quantile',
                                # 'Material.tau', 
                                # 'Material.par_tau', 
                                # 'Material.par_alpha', 
                                'Material.Covariance.log_nu',
                                # 'Material.Covariance.log_corrlen',
                                # 'Material.Structure.par_thickness'
                                'Material.noise_quantile'
                            ],
### Data
    'data_source'       :   'data3D_cell6.npy', ### 'surrogate', 'npy', 'image', '<filename>.npy'
    'input_folder'      :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
    'output_folder'     :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/cells/cell_6/',
}
test_seed = 0

MaterialClass = LatticeMaterial
# MaterialClass = GaussianMaterial
# MaterialClass = ParticlesMaterial

RF = MaterialClass(**config)
RF.calibration_regime = True

########################################################
# DATA
########################################################

if config['data_source'] == 'surrogate': ### Generate surrogate data
    RF0 = RF.copy()
    RF0.calibration_regime = True

    RF0.Covariance.nu = 0.75
    RF0.Covariance.corrlen = 0.05
    # # RF0.tau = 3
    RF0.alpha = 0.025
    RF0.Structure.thickness = 0.15
    RF0.noise_sparisity = 1.5
    # RF0.vf = 0.2

    if RF.ndim == 2:
        fig, axs = plt.subplots(1,2)
        axs[0].set_title('target')
        axs[1].set_title('initial')
        RF0.plot(ax=axs[0], show=False, seed=test_seed)
        RF.plot(ax=axs[1], show=True, seed=test_seed)
    else:
        RF0.save_vtk(config['output_folder']+'sample_target', seed=test_seed)

    Data = [ RF0.sample() for isample in range(10) ]

elif config['data_source'] == 'image': ### Data constructed from CT image
    Data = read_data3D(**config)

else: ### Load data from npy file
    if config['data_source'] == 'npy':
        filename = 'data3D.npy'
    else:
        filename = config['data_source']
    Data = load_data_from_npy(config['input_folder']+filename, downsample_shape=RF.Window.shape)
    # Data_full = np.load(config['input_folder']+filename)
    # n = np.min(Data_full.shape)
    # Data = torch.zeros([n]*Data_full.ndim, dtype=torch.float)
    # Data[:] = Data_full[:n,:n,:n]
    # # Data = interpolate(Data, RF.shape)
    RF.save_vtk(config['output_folder']+'sample_target', Sample=Data[0])
    # Data = [ Data ]

if RF.ndim == 3:
    RF.save_vtk(config['output_folder']+'sample_init', seed=test_seed)
    print('Samples saved to ', os.path.abspath(config['output_folder']))

########################################################

out = calibrate(MaterialClass, Data, **config)

RF = out.Model.Material
RF.calibration_regime = True
if RF.ndim == 3:
    RF.save_vtk(config['output_folder']+'sample_result', seed=test_seed)
    out.Model.Material.export_parameters(config['output_folder'] + 'inferred_parameters')

########################################################

### Plot (2D surrogate data)
if config['data_source'] == 'surrogate' and RF.ndim == 2:

    print('######################################')
    print('             RESULTS')
    print('######################################')
    with suppress(AttributeError): print('vf / vf_ref : ', [RF.vf, RF0.vf])
    with suppress(AttributeError): print('tau / tau_ref : ', [RF.tau.item(), RF0.tau.item()])
    with suppress(AttributeError): print('R / R_ref : ', [RF.Structure.radius, RF0.Structure.radius])
    print('nu / nu_ref : ', [RF.Covariance.nu.item(), RF0.Covariance.nu.item()])
    print('l  / l_ref  : ', [RF.Covariance.corrlen.data, RF0.Covariance.corrlen.data])
    print('t  = ', out.Model.quantile.item())
    print('S  = ', out.batch_size)

    fig, axs = plt.subplots(1,2)
    axs[0].set_title('target')
    axs[1].set_title('result')
    RF0.plot(ax=axs[0], show=False, seed=test_seed)
    RF.plot(ax=axs[1], show=True, seed=test_seed)
    
########################################################
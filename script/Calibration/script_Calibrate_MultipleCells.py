
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
from source.RMNet.Calibration.Calibration_LBFGS import save_as_csv

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
### Lattice
    'nCells'            :   1,
    'alpha'             :   0.1,
    'thickness'         :   0.3,
    'noise_sparsity'    :   0,
### Optimization
    'max_iter'          :   100,
    'SGD'               :   False,
    'mean_only'         :   True,
    'beta'              :   0.9,
    'history_size'      :   100,
    'curvature_eps'     :   0.01,
    'init_batch_size'   :   2,
    'lr'                :   1,
    'tol'               :   1.e-5,
    'line_search'       :   'Armijo', ### 'Armijo', 'None', 'Wolfe'
    'Powell_dumping'    :   False,
    'regularization'    :   1.e-4,
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
    # 'data_source'       :   [
    #                             'data3D_cell1.npy',
    #                             'data3D_cell2.npy',
    #                             'data3D_cell3.npy',
    #                         ],
    'input_folder'      :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
    'output_folder'     :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
}
test_seed = 0
ls_datafiles = [ config['input_folder']+'data3D_cell{0:d}.npy'.format(i+1) for i in range(12) ]
print('Using data from files:', ls_datafiles)

MaterialClass = LatticeMaterial

RF = MaterialClass(**config)
RF.calibration_regime = True
RF.save_vtk(config['output_folder']+'sample_init', seed=test_seed)
print('Samples saved to ', os.path.abspath(config['output_folder']))


"""
==================================================================================================================
Loop over the lattice cells
==================================================================================================================
"""


results = []

stored_batch_size = []
stored_grad_norm = []
stored_loss = []
stored_parameters_quantile = []
stored_parameters_thickness = []
stored_parameters_alpha = []
stored_parameters_nu = []
stored_parameters_corrlen = []
stored_parameters_noise_q = []

for icell, datafile in enumerate(ls_datafiles):

    Data = load_data_from_npy(datafile, downsample_shape=RF.Window.shape)
    RF.save_vtk(config['output_folder']+'sample_target_cell_' + str(icell), Sample=Data[0])

    out = calibrate(MaterialClass, Data, **config)
    out.Model.Material.export_parameters(config['output_folder'] + 'inferred_parameters_cell_' + str(icell))
    results.append(out)

    RF_i = out.Model.Material
    RF_i.calibration_regime = True
    RF_i.save_vtk(config['output_folder']+'sample_result_cell_' + str(icell), seed=test_seed)

    stored_batch_size.append(torch.tensor(out.stored_batch_size))
    stored_grad_norm.append(torch.tensor(out.stored_grad_norm))
    stored_loss.append(torch.tensor(out.stored_loss))
    stored_parameters_quantile.append(torch.tensor(out.stored_parameters_quantile))
    stored_parameters_thickness.append(torch.tensor(out.stored_parameters_thickness))
    stored_parameters_alpha.append(torch.tensor(out.stored_parameters_alpha))
    stored_parameters_nu.append(torch.tensor(out.stored_parameters_nu))
    stored_parameters_corrlen.append(torch.tensor(out.stored_parameters_corrlen))
    stored_parameters_noise_q.append(torch.tensor(out.stored_parameters_noise_q))


a = torch.nn.utils.rnn.pad_sequence(stored_batch_size)
# a = np.asarray(stored_batch_size, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_batch_size')

a = torch.nn.utils.rnn.pad_sequence(stored_grad_norm)[:,:,1]
# a = np.asarray(stored_grad_norm, dtype=float)[:,:,1]
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_grad_norm')

a = torch.nn.utils.rnn.pad_sequence(stored_loss)
# a = np.asarray(stored_loss, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_loss')

a = torch.nn.utils.rnn.pad_sequence(stored_parameters_quantile)
# a = np.asarray(stored_parameters_quantile, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_parameters_quantile')

a = torch.nn.utils.rnn.pad_sequence(stored_parameters_thickness)
# a = np.asarray(stored_parameters_thickness, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_parameters_thickness')

a = torch.nn.utils.rnn.pad_sequence(stored_parameters_nu)
# a = np.asarray(stored_parameters_nu, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_parameters_nu')

a = torch.nn.utils.rnn.pad_sequence(stored_parameters_corrlen)
# a = np.asarray(stored_parameters_corrlen, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_parameters_corrlen')

a = torch.nn.utils.rnn.pad_sequence(stored_parameters_noise_q)
# a = np.asarray(stored_parameters_noise_q, dtype=float)
save_as_csv(a, fileneame=config['output_folder']+'MCs_stored_parameters_noise_q')


########################################################

from math import pi
from source.RMNet.DataManager import read_data3D

i, j, k = 1, 1, 1
L, Lz = 0.315, 0.225
a, b = 0.32 - L/2 + i*L, 0.635 - L/2 + i*L
c, d = 0.32 - L/2 + j*L, 0.635 - L/2 + j*L
e, f = 0.365 + k*Lz, 0.585 + k*Lz

config = {
    'input_folder'      :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/200N/SlicesY_8bit/',
    'ext'               :   'tif',
    'output_folder'     :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
    'output_filename'   :   'data3D_cell12',
    'export_np'         :   True,
    'export_vtk'        :   True,
    'crop'              :   [a, b] + [c,d] + [e,f], ### ratio
    'threshold'         :   0.4, #0.4, ### ratio
}


read_data3D(**config)


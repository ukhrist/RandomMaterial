

import numpy as np
from pyevtk.hl import imageToVTK

"""
==================================================================================================================
Export to VTK
==================================================================================================================
"""

def exportVTK(FileName, cellData):
    shape = list(cellData.values())[0].shape
    ndim  = len(shape)
    N = min(shape)
    spacing = (1./N, 1./N, 1./N)

    if ndim==3:
        imageToVTK(FileName, cellData = cellData, spacing = spacing)

    elif ndim==2:
        cellData2D = {}
        for key in cellData.keys(): cellData2D[key] = np.expand_dims(cellData[key], axis=2)
        imageToVTK(FileName, cellData = cellData2D, spacing = spacing)

    else:
        raise Exception('Unsupported dimension {0:d}'.format(ndim))

    print('VTK exported to {0:s}'.format(FileName))



"""
==================================================================================================================
Testing gradients in PyTorch
==================================================================================================================
"""

def getBack(var_grad_fn):
        print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print(n[0])
                    print('Tensor with grad found:', tensor)
                    print(' - gradient:', tensor.grad)
                    print()
                except AttributeError as e:
                    getBack(n[0])
                    
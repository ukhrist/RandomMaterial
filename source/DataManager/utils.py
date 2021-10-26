
import numpy as np
from pyevtk.hl import imageToVTK



"""
==================================================================================================================
Export to VTK format
==================================================================================================================
"""


def exportVTK(FileName, cellData):
    shape   = list(cellData.values())[0].shape
    ndim    = len(shape)
    spacing = (1./min(shape), ) * 3

    if ndim==3:
        imageToVTK(FileName, cellData = cellData, spacing = spacing)

    elif ndim==2:
        cellData2D = {}
        for key in cellData.keys(): cellData2D[key] = np.expand_dims(cellData[key], axis=2)
        imageToVTK(FileName, cellData = cellData2D, spacing = spacing)

    else:
        raise Exception('Dimension must be 2 or 3.')


"""
==================================================================================================================
"""
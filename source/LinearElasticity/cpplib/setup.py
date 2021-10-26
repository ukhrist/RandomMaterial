
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
        Extension(  "cpplib_wrappers", 
                    ["cpplib_wrappers.pyx"],
                    language="c++",
                    include_dirs=[numpy.get_include()]
                    # libraries=['gomp'],
                    # extra_compile_args=['-fopenmp', '-std=c++17', '-Ofast', '-floop-parallelize-all', '-floop-nest-optimize', '-ftree-loop-distribution']
                )
    ])
)

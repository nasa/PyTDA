from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pytda_cython_tools", ["pytda_cython_tools.pyx"])]
)

# To install, do: python setup.py build_ext --inplace

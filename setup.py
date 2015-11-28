"""
Python Turbulence Detection Algorithm (PyTDA)
"""

import os
import sys
from distutils.core import setup
from setuptools import find_packages
from distutils.sysconfig import get_python_lib
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from distutils.core import setup

# - Pull the header into a variable
doclines = __doc__.split("\n")

VERSION = '1.1.1'

# - Set variables for setup
PACKAGES = ['pytda']
package_dir = {'': 'pytda'}

USE_CYTHON = True  # command line option, try-import, ...

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("pytda/pytda_cython_tools",
              ["pytda/pytda_cython_tools"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
      name='pytda',
      version=VERSION,
      url='http://github.com/nasa/PyTDA',
      author='Timothy Lang',
      author_email='timothy.j.lang@nasa.gov',
      description=doclines[0],
      packages=PACKAGES,
      classifiers=["""
        Development Status :: Beta,
        Programming Language :: Python",
        Topic :: Scientific/Engineering
        Topic :: Scientific/Engineering :: Atmospheric Science
        Operating System :: Unix
        Operating System :: POSIX :: Linux
        Operating System :: MacOS
        """],
      long_description="""Python Turbulence Detection Algorithm (PyTDA)""",
      ext_modules=extensions
)

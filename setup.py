# -*- coding: utf-8 -*-
"""GeostatTools: A geostatistical toolbox."""
from __future__ import division, absolute_import, print_function
import os
import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

DOCLINES = __doc__.split('\n')
README = open('README.md').read()

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Utilities',
]

EXT_MODULES = []
if USE_CYTHON:
    EXT_MODULES += cythonize(os.path.join('gstools', 'variogram',
                                          'estimator.pyx'))
else:
    EXT_MODULES += [Extension('gstools.variogram.estimator',
                              [os.path.join('gstools',
                                            'variogram',
                                            'estimator.c')],
                              include_dirs=[numpy.get_include()])]
# This is the important part. By setting this compiler directive, cython will
# embed signature information in docstrings. Sphinx then knows how to extract
# and use those signatures.
# python setup.py build_ext --inplace --> then sphinx build
for ext in EXT_MODULES:
    ext.cython_directives = {"embedsignature": True}
# version import not possible due to cython (separate __version__ in __init__)
VERSION = "0.3.6"

setup(
    name='gstools',
    version=VERSION,
    maintainer='Lennart Schueler',
    maintainer_email="lennart.schueler@ufz.de",
    description=DOCLINES[0],
    long_description=README,
    long_description_content_type="text/markdown",
    author="Lennart Schueler",
    author_email="lennart.schueler@ufz.de",
    url='https://github.com/LSchueler/GSTools',
    license='GPL - see LICENSE',
    classifiers=CLASSIFIERS,
    platforms=["Linux"],
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
    ],
    packages=find_packages(exclude=['tests*', 'docs*']),
    ext_modules=EXT_MODULES,
    include_dirs=[numpy.get_include()],
)

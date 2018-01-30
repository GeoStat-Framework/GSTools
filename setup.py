#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GeostatTools: A geostatistical toolbox."""
from __future__ import division, absolute_import, print_function


doclines = __doc__.split('\n')


classifiers = [
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

long_description = 'See `here <https://github.com/LSchueler/GSTools>`_ for more information.'

MAJOR      = 0
MINOR      = 3
MICRO      = 0
ISRELEASED = False
VERSION    = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []
if use_cython:
    ext_modules +=  cythonize(os.path.join('gstools', 'variogram',
                                           'estimator.pyx'))
else:
    ext_modules += [Extension('gstools.variogram.estimator',
        [os.path.join('gstools', 'variogram', 'estimator.c')])]

setup(
    name = 'gstools',
    version = VERSION,
    maintainer = 'Lennart Schueler',
    maintainer_email = "lennart.schueler@ufz.de",
    description = doclines[0],
    long_description = long_description,
    author = "Lennart Schueler, Falk Hesse",
    author_email = "lennart.schueler@ufz.de",
    url='https://github.com/LSchueler/GSTools',
    license = 'GPL - see LICENSE',
    classifiers = classifiers,
    platforms = ["Linux"],
    include_package_data = True,
    install_requires = ['numpy', 'scipy'],
    packages = find_packages(exclude=['tests*', 'docs*']),
    ext_modules = ext_modules,
)

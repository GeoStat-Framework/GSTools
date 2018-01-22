#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gstools

GeoStatTools

"""
doclines = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 5 - alpha
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Natural Language :: English
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development
Topic :: Utilities
"""

MAJOR      = 0
MINOR      = 2
MICRO      = 0
ISRELEASED = False
VERSION    = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

from setuptools import setup, find_packages

setup(
    name = 'gstools',
    version = VERSION,
    maintainer = 'Lennart Schueler',
    maintainer_email = "lennart.schueler (at) ufz (dot) de",
    description = doclines[0],
    long_description = open('README.md').read(),
    author = "Lennart Schueler, Falk Hesse",
    author_email = "lennart.schueler (at) ufz (dot) de",
    license = 'GPL - see LICENSE',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Linux"],
    include_package_data = True,
    install_requires = ['numpy', 'scipy'],
    packages = find_packages(exclude=['tests*', 'docs*']),
)

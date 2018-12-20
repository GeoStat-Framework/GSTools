# -*- coding: utf-8 -*-
"""GeostatTools: A geostatistical toolbox."""
from __future__ import division, absolute_import, print_function
import os
import codecs
import re
import logging
from distutils.errors import (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
)
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import numpy

logging.basicConfig()
log = logging.getLogger(__file__)

ext_errors = (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
    IOError,
)

HERE = os.path.abspath(os.path.dirname(__file__))


# version finder ##############################################################

def read(*parts):
    """read file data"""
    with codecs.open(os.path.join(HERE, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    """find version without importing module"""
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# cython handler ##############################################################

class BuildFailed(Exception):
    """Exeption for Cython build failed"""

    pass


def construct_build_ext(build_ext_base):
    """Construct a wrapper class for build_ext"""

    class WrappedBuildExt(build_ext_base):
        """This class allows C extension building to fail."""

        def run(self):
            """overridden run with try-except"""
            try:
                build_ext_base.run(self)
            except DistutilsPlatformError as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            """overridden build_extension with try-except"""
            try:
                build_ext_base.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)

    return WrappedBuildExt


# setup #######################################################################

try:
    from Cython.Build import cythonize
except ImportError:
    print("## GSTOOLS setup: cython not used")
    USE_CYTHON = False
else:
    print("## GSTOOLS setup: cython used")
    USE_CYTHON = True

DOCLINES = __doc__.split("\n")
README = open(os.path.join(HERE, "README.md")).read()

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Natural Language :: English",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Utilities",
]

EXT_MODULES = []
if USE_CYTHON:
    EXT_MODULES += cythonize(
        os.path.join("gstools", "variogram", "estimator.pyx")
    )
else:
    EXT_MODULES += [
        Extension(
            "gstools.variogram.estimator",
            [os.path.join("gstools", "variogram", "estimator.c")],
            include_dirs=[numpy.get_include()],
        )
    ]

# This is the important part. By setting this compiler directive, cython will
# embed signature information in docstrings. Sphinx then knows how to extract
# and use those signatures.
# python setup.py build_ext --inplace --> then sphinx build
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}
# version import not possible due to cython
# see: https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = find_version("gstools", "_version.py")

setup_kw = {
    "name": "gstools",
    "version": VERSION,
    "maintainer": "Lennart Schueler, Sebastian Mueller",
    "maintainer_email": "lennart.schueler@ufz.de, sebastian.mueller@ufz.de",
    "description": DOCLINES[0],
    "long_description": README,
    "long_description_content_type": "text/markdown",
    "author": "Lennart Schueler, Sebastian Mueller",
    "author_email": "lennart.schueler@ufz.de, sebastian.mueller@ufz.de",
    "url": "https://github.com/LSchueler/GSTools",
    "license": "GPL - see LICENSE",
    "classifiers": CLASSIFIERS,
    "platforms": ["Linux"],
    "include_package_data": True,
    "setup_requires": ["numpy>=1.8.2"],  # numpy imported in setup.py
    "install_requires": [
        "numpy>=1.8.2",
        "scipy>=0.19.1",
        "hankel>=0.3.6",
        "emcee",  # 2 and 3 should be compatible
        "pyevtk",
        "six",
    ],
    "packages": find_packages(exclude=["tests*", "docs*"]),
    "ext_modules": EXT_MODULES,
    "include_dirs": [numpy.get_include()],
}

cmd_classes = setup_kw.setdefault("cmdclass", {})

try:
    print("## GSTOOLS setup: try build with c code")
    # try building with c code :
    setup_kw["cmdclass"]["build_ext"] = construct_build_ext(build_ext)
    setup(**setup_kw)
except BuildFailed as ex:
    print("The C extension could not be compiled")
    log.warn(ex)
    log.warn("The C extension could not be compiled")

    ## Retry to install the module without C extensions :
    # Remove any previously defined build_ext command class.
    setup_kw["cmdclass"].pop("build_ext", None)
    cmd_classes.pop("build_ext", None)
    setup_kw.pop("ext_modules", None)

    # If this new 'setup' call doesn't fail, the module
    # will be successfully installed, without the C extensions
    setup(**setup_kw)
    print("## GSTOOLS setup: Plain-Python installation successful.")
else:
    print("## GSTOOLS setup: cython installation successful.")

# -*- coding: utf-8 -*-
"""GSTools: A geostatistical toolbox."""
import sys
import os
import glob
import tempfile
import subprocess

from distutils.errors import CompileError, LinkError
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

from setuptools import setup, find_packages, Distribution, Extension
from Cython.Build import cythonize
import numpy as np


HERE = os.path.abspath(os.path.dirname(__file__))


# openmp finder ###############################################################
# This code is adapted for a large part from the scikit-learn openmp helpers,
# which can be found at:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/_build_utils/openmp_helpers.py


CCODE = """
#include <omp.h>
#include <stdio.h>
int main(void) {
#pragma omp parallel
printf("nthreads=%d\\n", omp_get_num_threads());
return 0;
}
"""


def get_openmp_flag(compiler):
    """Get the compiler dependent openmp flag."""
    if hasattr(compiler, "compiler"):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
        return ["/Qopenmp"]
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin" and ("icc" in compiler or "icl" in compiler):
        return ["-openmp"]
    if sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
        # export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
        return []
    # Default flag for GCC and clang:
    return ["-fopenmp"]


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run."""
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            # Write test program
            with open("test_openmp.c", "w") as cfile:
                cfile.write(CCODE)
            os.mkdir("objects")
            # Compile, test program
            openmp_flags = get_openmp_flag(ccompiler)
            ccompiler.compile(
                ["test_openmp.c"],
                output_dir="objects",
                extra_postargs=openmp_flags,
            )
            # Link test program
            extra_preargs = os.getenv("LDFLAGS", None)
            if extra_preargs is not None:
                extra_preargs = extra_preargs.split(" ")
            else:
                extra_preargs = []
            objects = glob.glob(
                os.path.join("objects", "*" + ccompiler.obj_extension)
            )
            ccompiler.link_executable(
                objects,
                "test_openmp",
                extra_preargs=extra_preargs,
                extra_postargs=openmp_flags,
            )
            # Run test program
            output = subprocess.check_output("./test_openmp")
            output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
            # Check test program output
            if "nthreads=" in output[0]:
                nthreads = int(output[0].strip().split("=")[1])
                openmp_supported = len(output) == nthreads
            else:
                openmp_supported = False
                openmp_flags = []
        except (CompileError, LinkError, subprocess.CalledProcessError):
            openmp_supported = False
            openmp_flags = []
        finally:
            os.chdir(HERE)
    return openmp_supported, openmp_flags


# openmp ######################################################################


USE_OPENMP = bool("--openmp" in sys.argv)

if USE_OPENMP:
    # just check if wanted
    CAN_USE_OPENMP, FLAGS = check_openmp_support()
    if CAN_USE_OPENMP:
        print("## GSTOOLS setup: OpenMP found.")
        print("## OpenMP flags:", FLAGS)
    else:
        print("## GSTOOLS setup: OpenMP not found.")
else:
    print("## GSTOOLS setup: OpenMP not wanted by the user.")
    FLAGS = []


# add the "--openmp" to the global options
# enables calles like:
# python3 setup.py --openmp build_ext --inplace
# pip install --global-option="--openmp" gstools
class MPDistribution(Distribution):
    """Distribution with --openmp as global option."""

    global_options = Distribution.global_options + [
        ("openmp", None, "Flag to use openmp in the build")
    ]


# cython extensions ###########################################################


CY_MODULES = []
CY_MODULES.append(
    Extension(
        "gstools.field.summator",
        [os.path.join("gstools", "field", "summator.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=FLAGS,
        extra_link_args=FLAGS,
    )
)
CY_MODULES.append(
    Extension(
        "gstools.variogram.estimator",
        [os.path.join("gstools", "variogram", "estimator.pyx")],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=FLAGS,
        extra_link_args=FLAGS,
    )
)
CY_MODULES.append(
    Extension(
        "gstools.krige.krigesum",
        [os.path.join("gstools", "krige", "krigesum.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=FLAGS,
        extra_link_args=FLAGS,
    )
)
EXT_MODULES = cythonize(CY_MODULES)  # annotate=True

# This is an important part. By setting this compiler directive, cython will
# embed signature information in docstrings. Sphinx then knows how to extract
# and use those signatures.
# python setup.py build_ext --inplace --> then sphinx build
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}

# setup #######################################################################

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    README = f.read()
with open(os.path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
    REQ = f.read().splitlines()
with open(os.path.join(HERE, "requirements_setup.txt"), encoding="utf-8") as f:
    REQ_SETUP = f.read().splitlines()
with open(os.path.join(HERE, "requirements_test.txt"), encoding="utf-8") as f:
    REQ_TEST = f.read().splitlines()
with open(
    os.path.join(HERE, "docs", "requirements_doc.txt"), encoding="utf-8"
) as f:
    REQ_DOC = f.read().splitlines()

REQ_DEV = REQ_SETUP + REQ_TEST + REQ_DOC

DOCLINE = __doc__.split("\n")[0]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Natural Language :: English",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Utilities",
]

setup(
    name="gstools",
    description=DOCLINE,
    long_description=README,
    long_description_content_type="text/markdown",
    maintainer="Lennart Schueler, Sebastian Mueller",
    maintainer_email="info@geostat-framework.org",
    author="Lennart Schueler, Sebastian Mueller",
    author_email="info@geostat-framework.org",
    url="https://github.com/GeoStat-Framework/GSTools",
    license="LGPLv3",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
    python_requires=">=3.5",
    use_scm_version={
        "relative_to": __file__,
        "write_to": "gstools/_version.py",
        "write_to_template": "__version__ = '{version}'",
        "local_scheme": "no-local-version",
        "fallback_version": "0.0.0.dev0",
    },
    setup_requires=REQ_SETUP,
    install_requires=REQ,
    extras_require={
        "plotting": ["pyvista", "matplotlib"],
        "doc": REQ_DOC,
        "test": REQ_TEST,
        "dev": REQ_DEV,
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    ext_modules=EXT_MODULES,
    include_dirs=[np.get_include()],
    distclass=MPDistribution,
)

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

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


HERE = os.path.abspath(os.path.dirname(__file__))


# openmp finder ###############################################################
# This code is adapted for a large part from the scikit-learn openmp_helpers.py
# which can be found at:
# https://github.com/scikit-learn/scikit-learn/blob/0.24.0/sklearn/_build_utils


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


# you can set GSTOOLS_BUILD_PARALLEL=0 or GSTOOLS_BUILD_PARALLEL=1
GS_PARALLEL = os.getenv("GSTOOLS_BUILD_PARALLEL")
USE_OPENMP = bool(int(GS_PARALLEL)) if GS_PARALLEL else False

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

# embed signatures for sphinx
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}


# setup #######################################################################


setup(ext_modules=EXT_MODULES, include_dirs=[np.get_include()])

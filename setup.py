# -*- coding: utf-8 -*-
"""GSTools: A geostatistical toolbox."""
import os

import numpy as np
from Cython.Build import cythonize
from extension_helpers import add_openmp_flags_if_available
from setuptools import Extension, setup

# cython extensions
CY_FILES = [
    {"name": "gstools.field.summator", "language": "c"},
    {"name": "gstools.variogram.estimator", "language": "c++"},
    {"name": "gstools.krige.krigesum", "language": "c"},
]
CY_MODULES = [
    Extension(
        name=file["name"],
        sources=[os.path.join("src", *file["name"].split(".")) + ".pyx"],
        language=file["language"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for file in CY_FILES
]
# you can set GSTOOLS_BUILD_PARALLEL=0 or GSTOOLS_BUILD_PARALLEL=1
if int(os.getenv("GSTOOLS_BUILD_PARALLEL", "0")):
    for mod in CY_MODULES:
        openmp_flags_added = add_openmp_flags_if_available(mod)
    if openmp_flags_added:
        print("## GSTOOLS setup: OpenMP found.")
    else:
        print("## GSTOOLS setup: OpenMP not found.")
else:
    print("## GSTOOLS setup: OpenMP not wanted by the user.")

# setup - do not include package data to ignore .pyx files in wheels
setup(ext_modules=cythonize(CY_MODULES), include_package_data=False)

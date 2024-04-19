"""GSTools: A geostatistical toolbox."""

import os

import numpy as np
from Cython.Build import cythonize
from extension_helpers import add_openmp_flags_if_available
from setuptools import Extension, setup

# cython extensions
CY_MODULES = [
    Extension(
        name=f"gstools.{ext}",
        sources=[os.path.join("src", "gstools", *ext.split(".")) + ".pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for ext in ["field.summator", "variogram.estimator", "krige.krigesum"]
]
# you can set GSTOOLS_BUILD_PARALLEL=0 or GSTOOLS_BUILD_PARALLEL=1
open_mp = False
if int(os.getenv("GSTOOLS_BUILD_PARALLEL", "0")):
    added = [add_openmp_flags_if_available(mod) for mod in CY_MODULES]
    if any(added):
        open_mp = True
    print(f"## GSTools setup: OpenMP used: {open_mp}")
else:
    print("## GSTools setup: OpenMP not wanted by the user.")

# setup - do not include package data to ignore .pyx files in wheels
setup(
    ext_modules=cythonize(CY_MODULES, compile_time_env={"OPENMP": open_mp}),
    include_package_data=False,
)

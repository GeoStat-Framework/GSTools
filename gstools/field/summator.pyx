# cython: language_level=2
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from libc.math cimport sqrt
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


# variables needed:
# _cov_sample
# positions
# random vars _z_1, _z_2
def summate():
    pass

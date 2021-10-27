# -*- coding: utf-8 -*-
"""GSTools: A geostatistical toolbox."""
import os

import numpy as np
from setuptools import setup
import sys
import site

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

HERE = os.path.abspath(os.path.dirname(__file__))

setup(include_dirs=[np.get_include()])

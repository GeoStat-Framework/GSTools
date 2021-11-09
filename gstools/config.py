# -*- coding: utf-8 -*-
"""
GStools subpackage providing global variables.

.. currentmodule:: gstools.config

"""
# pylint: disable=W0611
try:
    import gstools_core

    USE_RUST = True #pragma: no cover
except ImportError:
    USE_RUST = False

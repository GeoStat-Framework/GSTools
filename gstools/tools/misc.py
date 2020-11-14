# -*- coding: utf-8 -*-
"""
GStools subpackage providing miscellaneous tools.

.. currentmodule:: gstools.tools.misc

The following functions are provided

.. autosummary::
   list_format
"""


def list_format(lst, prec):
    """Format a list of floats."""
    return "[{}]".format(", ".join(f"{x:.{prec}}" for x in lst))

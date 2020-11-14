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
    return "[{}]".format(
        ", ".join("{x:.{p}}".format(x=x, p=prec) for x in lst)
    )

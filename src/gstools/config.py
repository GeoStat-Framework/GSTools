"""
GStools subpackage providing global variables.

.. currentmodule:: gstools.config

"""

NUM_THREADS = None

# pylint: disable=W0611
try:  # pragma: no cover
    import gstools_core

    _GSTOOLS_CORE_AVAIL = True
    USE_GSTOOLS_CORE = True
except ImportError:
    _GSTOOLS_CORE_AVAIL = False
    USE_GSTOOLS_CORE = False

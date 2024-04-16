"""
GStools subpackage providing global variables.

.. currentmodule:: gstools.config

"""

NUM_THREADS = None

# pylint: disable=W0611
try:  # pragma: no cover
    import gstools_core

    USE_RUST = True
except ImportError:
    USE_RUST = False

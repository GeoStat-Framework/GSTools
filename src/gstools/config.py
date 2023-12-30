"""
GStools subpackage providing global variables.

.. currentmodule:: gstools.config

"""
NUM_THREADS = 1

# pylint: disable=W0611
try:  # pragma: no cover
    import gstools_core

    USE_RUST = True
except ImportError:
    USE_RUST = False

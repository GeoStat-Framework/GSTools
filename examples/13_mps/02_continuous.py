r"""
Continuous variables and distance metrics
-----------------------------------------

Direct Sampling is not limited to categorical facies: with ``categorical=False``
it simulates continuous variables (permeability, porosity, elevation, ...). The
``distance`` argument then selects how two patterns are compared:

* ``"l1"`` / ``"l2"`` — Manhattan / Euclidean distance on the raw values
  (Mariethoz et al., 2010, Eq. 6 / Eq. 4).
* ``"variation"`` — compares only the *relative* variations within a pattern
  (Eq. 9), tolerating a locally varying mean. Useful for non-stationary data.

Here we use a smooth continuous training image and a small acceptance
``threshold`` (standard DS mode) to allow approximate matches.

.. note::

    The acceptance ``threshold`` is **not comparable across metrics**: the
    ``"variation"`` distance is normalised by ``2·d_max``, so the same value is
    stricter than it would be for ``"l1"``/``"l2"``. Re-tune it when you switch
    the distance metric.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# A smooth, continuous synthetic training image.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = np.sin(gx / 6.0) * np.cos(gy / 8.0)

###############################################################################
# Build a continuous training image with the Euclidean (``"l2"``) distance.

ti = gs.TrainingImage(ti_data, categorical=False, distance="l2")
print(ti)

###############################################################################
# Simulate. ``threshold=0.03`` accepts the first pattern within that distance
# (standard DS), which is faster than the exhaustive best-candidate search for
# continuous variables.

ds = gs.DirectSampling(ti, n_neighbors=12, scan_fraction=0.3, threshold=0.03)
field = ds([np.arange(32, dtype=float)] * 2, seed=3)

###############################################################################
# Plot. The realization reproduces the smooth wavy structure of the TI without
# copying it. A shared colour scale makes the comparison fair.

vmin, vmax = ti_data.min(), ti_data.max()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
im = ax0.imshow(ti_data, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
ax0.set_title("Training image (continuous)")
ax1.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
ax1.set_title("DS realization")
fig.colorbar(im, ax=(ax0, ax1), shrink=0.7)

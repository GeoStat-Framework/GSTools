r"""
Different scales
================

Besides the length-scale, there are many other ways of characterizing a certain
scale of a covariance model. We provide two common scales with the covariance
model.

Integral scale
--------------

The `integral scale <https://en.wikipedia.org/wiki/Integral_length_scale>`_
of a covariance model is calculated by:

.. math:: I = \int_0^\infty \rho(r) dr

You can access it by:
"""
import gstools as gs

model = gs.Stable(dim=3, var=2.0, len_scale=10)
print("Main integral scale:", model.integral_scale)
print("All integral scales:", model.integral_scale_vec)


###############################################################################
# You can also specify integral length scales like the ordinary length scale,
# and len_scale/anis will be recalculated:

model = gs.Stable(dim=3, var=2.0, integral_scale=[10, 4, 2])
print("Anisotropy ratios:", model.anis)
print("Main length scale:", model.len_scale)
print("All length scales:", model.len_scale_vec)
print("Main integral scale:", model.integral_scale)
print("All integral scales:", model.integral_scale_vec)


###############################################################################
# Percentile scale
# ----------------
#
# Another scale characterizing the covariance model, is the percentile scale.
# It is the distance, where the normalized
# variogram reaches a certain percentage of its sill.

model = gs.Stable(dim=3, var=2.0, len_scale=10)
per_scale = model.percentile_scale(0.9)
int_scale = model.integral_scale
len_scale = model.len_scale
print("90% Percentile scale:", per_scale)
print("Integral scale:", int_scale)
print("Length scale:", len_scale)

###############################################################################
# .. note::
#
#    The nugget is neglected by the percentile scale.
#
#
# Comparison
# ----------

ax = model.plot()
ax.axhline(1.8, color="k", label=r"90% percentile")
ax.axvline(per_scale, color="k", linestyle="--", label=r"90% percentile scale")
ax.axvline(int_scale, color="k", linestyle="-.", label=r"integral scale")
ax.axvline(len_scale, color="k", linestyle=":", label=r"length scale")
ax.legend()

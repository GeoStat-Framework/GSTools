r"""
A Layered Field
---------------

In this first example we build a simple layered subsurface in a vertical
2D cross-section. The interfaces between the units are 1D random *height functions*
:math:`z = f_i(x)`.

Instead of modelling the interfaces directly, a :any:`LayerStack` models
the thickness of each layer as a log-normal field

.. math::
   t_i(x) = \exp(g_i(x)), \quad g_i \sim \mathcal{N},

and stacks them on top of a base :math:`f_0(x)`:

.. math::
   f_{i+1}(x) = f_i(x) + t_i(x).

Since all thicknesses are positive, the interfaces never cross.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# The cross-section spans 200 m laterally and 40 m vertically.
#
# The Gaussian parameters of the log-thickness fields are easiest derived
# from a mean thickness and a coefficient of variation (CV) with
# :any:`LogLink.from_moments` (it also accepts the standard deviation).

x = np.linspace(0, 200, 200)
z = np.linspace(-10, 30, 150)

layers = []
for mean_thickness, cv, len_scale in [
    (8, 0.3, 40),
    (5, 0.6, 25),
    (7, 0.4, 60),
]:
    mu_g, var_g = gs.geometry.LogLink.from_moments(mean_thickness, cv=cv)
    model = gs.Gaussian(dim=1, var=var_g, len_scale=len_scale)
    layers.append(gs.Layer.from_model(model, mean=mu_g))

###############################################################################
# The base of the stack can be a constant, a function of the lateral
# coordinates or a random field itself. Here we use an undulating base.
# The master ``seed`` of the stack derives distinct seeds for all fields.


def base(x):
    """Undulating base of the stack."""
    return -2.0 + 3.0 * np.sin(x / 30.0)


stack = gs.LayerStack(layers, base=base, dim=2, seed=20170519)
print(stack)

###############################################################################
# A :any:`FaciesField` renders the integer labels of the geometry on any
# mesh. Below the base, the stack assigns ``below_label=-1`` and above the
# top ``above_label = n_layers = 3``.

fac = gs.FaciesField(stack)
labels = fac.structured((x, z))
print(np.unique(labels))

###############################################################################
# The interface elevations can be evaluated directly on the lateral
# positions. They are the exact boundaries of the labels.

ifaces = stack.interfaces(x)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.pcolormesh(x, z, labels.T, cmap="viridis", shading="nearest")
for iface in ifaces:
    ax.plot(x, iface, color="k", lw=1)
ax.set_xlabel("x / m")
ax.set_ylabel("z / m")
ax.set_title("Log-thickness layer stack")
fig.tight_layout()
plt.show()

###############################################################################
# The thicknesses themselves are available as well and are log-normally
# distributed with the given mean and CV.

thick = stack.thicknesses(x)
print("mean thicknesses:", thick.mean(axis=1).round(2))

###############################################################################
# The same code works in 3D: use models with ``dim=2`` for the layers, a
# stack with ``dim=3`` and evaluate the facies field on
# ``fac.structured((x, y, z))``. The vertical axis is always the last one.

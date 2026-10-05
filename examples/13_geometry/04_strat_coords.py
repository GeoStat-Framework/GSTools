r"""
Properties in Stratigraphic Coordinates
---------------------------------------

Properties like porosity or hydraulic conductivity usually follow the
layering within a unit: sediments are correlated along the bedding and
not along horizontal planes. Height functions provide *stratigraphic
coordinates* for free:

.. math::
   w = \frac{z - f_i(x)}{f_{i+1}(x) - f_i(x)} \in [0, 1),

with :math:`i` being the unit index. A property field evaluated on
:math:`(x, w)` instead of :math:`(x, z)` then follows the interfaces
(proportional gridding).
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# We use a stack of three layers with strongly varying thicknesses.

x = np.linspace(0, 200, 200)
z = np.linspace(-5, 42, 150)

layers = []
thickness_moments = [(10, 0.4), (8, 0.5), (10, 0.4)]
for mean_thickness, cv in thickness_moments:
    mu_g, var_g = gs.geometry.LogLink.from_moments(mean_thickness, cv=cv)
    model = gs.Gaussian(dim=1, var=var_g, len_scale=40)
    layers.append(gs.Layer.from_model(model, mean=mu_g))


def base(x):
    """Tilted and folded base of the stack."""
    return 0.03 * x + 3.0 * np.sin(x / 25.0)


stack = gs.LayerStack(layers, base=base, dim=2, seed=20170519)

###############################################################################
# The stratigraphic coordinates are evaluated on the structured grid. The
# result is flattened in the same order as :any:`generate_grid`.

idx, w = stack.strat_coords((x, z), mesh_type="structured")
x_pos, z_pos = gs.generate_grid((x, z))

###############################################################################
# Each unit gets its own porosity field with its own mean. The fields
# are strongly anisotropic, with a long correlation along the layering.
# To keep the length scales in meters, the relative coordinate ``w`` is
# scaled by a typical unit thickness of 10 m. For comparison we also
# evaluate the same fields on the plain :math:`(x, z)` coordinates.

por_strat = np.full(x_pos.size, np.nan)
por_plain = np.full(x_pos.size, np.nan)
for i, mean in enumerate([0.25, 0.15, 0.3]):
    model = gs.Exponential(dim=2, var=0.03**2, len_scale=[50, 1.5])
    srf = gs.SRF(model, mean=mean, seed=i)
    sel = idx == i
    por_strat[sel] = srf((x_pos[sel], 10.0 * w[sel]))
    por_plain[sel] = srf((x_pos[sel], z_pos[sel]))

###############################################################################
# Within each unit, the porosity in stratigraphic coordinates follows the
# interfaces, while the one in plain coordinates is cut off by them.

ifaces = stack.interfaces(x)
fig, axes = plt.subplots(
    1, 2, figsize=(12, 3.8), sharey=True, layout="constrained"
)
fields = [por_strat, por_plain]
titles = ["stratigraphic coordinates (x, w)", "plain coordinates (x, z)"]
for i in range(2):
    ax = axes[i]
    por = fields[i]
    im = ax.pcolormesh(
        x,
        z,
        por.reshape(len(x), len(z)).T,
        vmin=0.1,
        vmax=0.35,
        shading="nearest",
    )
    for iface in ifaces:
        ax.plot(x, iface, color="k", lw=0.8)
    ax.set_ylim(z[0], z[-1])
    ax.set_title(titles[i])
    ax.set_xlabel("x / m")
axes[0].set_ylabel("z / m")
fig.colorbar(im, ax=axes, label="porosity")
plt.show()

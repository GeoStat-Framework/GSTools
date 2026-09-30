r"""
Erosion and Onlap
-----------------

A single stack describes a conformable sequence of units. Real
stratigraphy often consists of several such *groups*, separated by
unconformities. A :any:`StratColumn` combines groups, ordered from oldest
to youngest, by one relation per contact:

- ``"erode"``: the younger group cuts into the older ones and removes
  everything above its own top,
- ``"onlap"``: the younger group only fills the free space above the
  older ones, which are kept.

Labels are assigned consecutively across all groups, oldest first.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

x = np.linspace(0, 200, 200)
z = np.linspace(-5, 35, 150)

###############################################################################
# The older group consists of three layers on a flat base.


def log_layer(mean_thickness, std, len_scale):
    """Log-thickness layer from mean thickness and standard deviation."""
    mu_g, var_g = gs.geometry.LogLink.from_moments(mean_thickness, std=std)
    model = gs.Gaussian(dim=1, var=var_g, len_scale=len_scale)
    return gs.Layer.from_model(model, mean=mu_g)


old = gs.LayerStack(
    [log_layer(5, 1.5, 50), log_layer(4, 1.6, 40), log_layer(6, 1.8, 60)],
    base=0.0,
    dim=2,
    seed=20170519,
)

###############################################################################
# The younger group of two layers is deposited on an erosional surface
# with a valley incised into the older group.


def valley(x):
    """Erosion surface with an incised valley."""
    return 12.0 - 8.0 * np.exp(-(((x - 110.0) / 35.0) ** 2))


young = gs.LayerStack(
    [log_layer(4, 1.2, 30), log_layer(4, 1.2, 30)],
    base=valley,
    dim=2,
    seed=19970221,
)

###############################################################################
# Now we combine both groups once with an erosive and once with an
# onlapping contact. With erosion, the younger units fill the valley.
# With onlap, the older units are kept and the younger ones only appear
# where they lie above the older group.

erode = gs.StratColumn([old, young], relations=["erode"])
onlap = gs.StratColumn([old, young], relations=["onlap"])
print(erode, erode.relations)

###############################################################################
# Finally, a topography cuts the erosive column. A :any:`HalfSpace` is the
# region above (or below) a lateral surface; here a random topography
# assigns the air label on top of the column, which is achieved by a
# higher ``priority`` in a :any:`Composite`. The same result can be
# obtained with the ``topography`` argument of :any:`StratColumn`.

air = 5  # labels 0..4 are the units of both groups
topo = gs.SRF(gs.Gaussian(dim=1, var=6, len_scale=30), mean=17, seed=1)
cut = gs.geometry.Composite(
    [erode, gs.HalfSpace(topo, dim=2, label=air, priority=1)]
)

###############################################################################
# All three geometries are rendered with a :any:`FaciesField`.

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
geometries = [erode, onlap, cut]
titles = ["erode", "onlap", "erode + topography"]
for i in range(3):
    ax = axes[i]
    labels = gs.FaciesField(geometries[i]).structured((x, z))
    ax.pcolormesh(
        x, z, labels.T, cmap="viridis", vmin=-1, vmax=air, shading="nearest"
    )
    ax.set_title(titles[i])
    ax.set_xlabel("x / m")
axes[0].set_ylabel("z / m")
fig.tight_layout()
plt.show()

###############################################################################
# Groups can also be :any:`SurfaceStack` objects or any other
# :any:`Region`, and more than two groups can be chained with mixed
# relations, e.g. ``relations=["erode", "onlap"]``.

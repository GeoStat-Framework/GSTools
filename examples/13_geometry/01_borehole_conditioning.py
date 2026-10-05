r"""
Conditioning on Boreholes
-------------------------

Height functions can be conditioned *exactly* on borehole data, since
every interface is just a random field on the lateral domain. Each
borehole reduces to data on single layers:

- two consecutive contacts give a thickness (an equality),
- a hole ending inside a unit gives a lower bound on its thickness,
- a zero thickness means the unit is absent (an upper bound),
- the lowest contact gives an elevation of the base.

Equalities are honoured by kriging (conditioned fields with
``cond_method="error"``), inequalities are first imputed by a Gibbs
sampler of the truncated Gaussian distribution and then honoured
exactly as well.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# We set up a stack of three layers in a 2D cross-section. To condition
# the base on the lowest contacts too, the base is a random field with
# a mean elevation of 0 m.
#
# A pinch-out is the upper bound :math:`t \le t_\mathrm{absent}` on the
# thickness, i.e. :math:`g \le \log(t_\mathrm{absent})` for the
# log-thickness. The default ``absent=1e-6`` lies far in the tail of
# the log-thickness distribution, which would pull the whole field down
# around the well. We therefore treat units thinner than 10 cm as absent
# by passing a :any:`LogLink` with ``absent=0.1`` and use a
# :any:`Matern` model, which extrapolates such strong gradients more
# gently than a :any:`Gaussian` model.

x = np.linspace(0, 200, 200)
z = np.linspace(-10, 32, 150)

base = gs.SRF(gs.Gaussian(dim=1, var=4, len_scale=50))
link = gs.geometry.LogLink(absent=0.1)
layers = []
thickness_moments = [(8, 0.3), (5, 0.8), (7, 0.4)]
for mean_thickness, cv in thickness_moments:
    mu_g, var_g = gs.geometry.LogLink.from_moments(mean_thickness, cv=cv)
    model = gs.Matern(dim=1, var=var_g, len_scale=40)
    layers.append(gs.Layer.from_model(model, mean=mu_g, link=link))
stack = gs.LayerStack(layers, base=base, dim=2, seed=19970221)

###############################################################################
# The well logs give the depths of the four interfaces, measured down from the
# collar (the top of the hole), ordered bottom to top with the base being
# interface 0. NaN means the contact was not identified or not reached:
#
# - well ``W3`` ends at a total depth (``td``) inside the lowest unit,
#   so this unit is at least as thick as the drilled part,
# - in well ``W5`` interfaces 1 and 2 coincide, i.e. the middle unit
#   pinches out there.

collar_z = 32.0
tops = {
    "W1": [33.0, 24.0, 19.0, 12.0],
    "W2": [34.0, 25.0, 17.0, 10.0],
    "W3": [np.nan, 23.0, 15.0, 9.0],
    "W4": [31.0, 22.0, 18.0, 11.0],
    "W5": [30.0, 20.0, 20.0, 13.0],
    "W6": [32.0, 24.0, 21.0, 14.0],
}
well_x = {"W1": 15, "W2": 50, "W3": 85, "W4": 115, "W5": 150, "W6": 185}
collar = {w: (well_x[w], collar_z) for w in tops}
td = {"W3": 27.0}

data, rejected = gs.geometry.from_well_logs(
    tops, layers=3, collar=collar, td=td
)
print(data)

###############################################################################
# The top of the lowest unit in ``W3`` is an elevation, which would
# constrain the *sum* of the base elevation and the first thickness.
# Since this can't be reduced to a single-layer datum, it is rejected
# with a warning (and returned in ``rejected``). The thickness bound
# from the total depth and the thicknesses above are kept.
# Use a :any:`SurfaceStack` if contact elevations are the primary data.

for rej in rejected:
    print(rej)

###############################################################################
# Now we condition the stack. Every layer and the base get their own
# conditioned field. The residuals of the current realization are zero up
# to round-off, for equalities as well as for inequalities.

stack.condition(data, seed=1)
print("max residual:", stack.residuals(data).max())

###############################################################################
# To draw further realizations, we reset the field seeds and redraw the
# Gibbs imputation of the inequality data. The facies are rendered with a
# :any:`FaciesField`.

fac = gs.FaciesField(stack)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for i in range(3):
    ax = axes[i]
    seed = i + 1
    stack.reset(seed=seed)
    stack.resample_inequalities(seed=seed)
    print(f"seed {seed}, max residual:", stack.residuals(data).max())
    labels = fac.structured((x, z))
    ax.pcolormesh(x, z, labels.T, cmap="viridis", shading="nearest")
    for iface in stack.interfaces(x):
        ax.plot(x, iface, color="k", lw=0.8)
    for well, md in tops.items():
        wx = well_x[well]
        bottom = collar_z - td.get(well, np.nanmax(md))
        ax.plot([wx, wx], [collar_z, bottom], color="w", lw=2)
        ax.scatter(np.full(4, wx), collar_z - np.array(md), c="r", s=12)
    ax.set_title(f"realization {seed}")
    ax.set_xlabel("x / m")
axes[0].set_ylabel("z / m")
fig.tight_layout()
plt.show()

###############################################################################
# All realizations honour the logs: the contacts (red) lie exactly on the
# interfaces, the middle unit vanishes at ``W5`` and the lowest unit in
# ``W3`` extends below the bottom of the hole.
#
# The only exception are the contact elevations in ``W3``: since the hole
# does not reach the base, only the *thicknesses* of the upper units are
# honoured there, while the absolute elevation of the contacts is free.
# This is the price of the thickness parameterisation. If contact
# elevations matter more than thicknesses, use a :any:`SurfaceStack`, as
# shown in a later example.

r"""
Surfaces from Surface Points
----------------------------

Instead of layer thicknesses, a :any:`SurfaceStack` models the elevation
of every interface :math:`h_0, \dots, h_K` as an independent random field.
Crossing surfaces are resolved by a cumulative maximum (onlap)

.. math::
   f_0 = h_0, \quad f_i = \max(f_{i-1}, h_i),

or a cumulative minimum from the top (erode). Every contact elevation is
then an exact equality on a single surface, which makes this class the
natural choice for surface-points tables, where contacts are
given as ``x, [y,] z, surface``.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

x = np.linspace(0, 200, 200)
z = np.linspace(-15, 30, 150)

###############################################################################
# The four surfaces bound three units. Each surface has its own mean
# elevation and variability.

names = ["base", "top A", "top B", "top C"]
means = [0.0, 8.0, 14.0, 20.0]
surfaces = []
for i in range(len(names)):
    model = gs.Matern(dim=1, var=6, len_scale=50)
    surfaces.append(gs.Surface.from_model(model, mean=means[i], name=names[i]))
stack = gs.SurfaceStack(surfaces, relation="onlap", dim=2, seed=20170519)

###############################################################################
# The surface-points table lists one contact per row. Contacts sharing
# a well id belong to one borehole, which lets us encode:
#
# - well ``W2`` ends at an elevation of 1 m (``td``) without reaching the
#   base, so the base must lie below the bottom of the hole,
# - in well ``W4`` "top A" and "top B" coincide, i.e. unit B is absent,
# - ``O1`` is a single outcrop point of "top C".

table = [
    # well, x, z, surface
    ("W1", 20.0, -2.0, "base"),
    ("W1", 20.0, 7.0, "top A"),
    ("W1", 20.0, 13.0, "top B"),
    ("W1", 20.0, 21.0, "top C"),
    ("W2", 70.0, 9.0, "top A"),
    ("W2", 70.0, 15.0, "top B"),
    ("W2", 70.0, 19.0, "top C"),
    ("W3", 120.0, 1.0, "base"),
    ("W3", 120.0, 6.0, "top A"),
    ("W3", 120.0, 12.0, "top B"),
    ("W3", 120.0, 22.0, "top C"),
    ("W4", 170.0, -1.0, "base"),
    ("W4", 170.0, 10.0, "top A"),
    ("W4", 170.0, 10.0, "top B"),
    ("W4", 170.0, 18.0, "top C"),
    ("O1", 195.0, 17.0, "top C"),
]
well = np.array([row[0] for row in table])
px = np.array([row[1] for row in table])
pz = np.array([row[2] for row in table])
psurf = np.array([row[3] for row in table])

data, rejected = gs.geometry.from_surface_points(
    px, None, pz, psurf, surfaces=names, well=well, td={"W2": 1.0}
)
print(data)

###############################################################################
# Under onlap, the lowest coincident surface carries the equality and the ones
# above get an upper bound (they may lie below and are then covered by the
# onlap). The base in ``W2`` gets the upper bound from the bottom of the hole.
# Surfaces below the lowest contact of a well are bound by it, so from outcrop
# ``O1`` we also know that the lower surfaces lie below 17 m there. These
# inequalities are imputed by a Gibbs sampler when conditioning.

for i in range(len(data)):
    print(
        f"{data.well[i]:>3} surface {data.layer[i]}: "
        f"value={data.value[i]:5.1f}, "
        f"bounds=({data.lower[i]:5.1f}, {data.upper[i]:5.1f})"
    )

stack.condition(data, seed=1)
print("max residual:", stack.residuals(data).max())

###############################################################################
# Now we plot three conditioned realizations together with the surface
# points and the drilled part of ``W2``.

fac = gs.FaciesField(stack)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for i in range(3):
    ax = axes[i]
    seed = i + 1
    stack.reset(seed=seed)
    stack.resample_inequalities(seed=seed)
    labels = fac.structured((x, z))
    ax.pcolormesh(x, z, labels.T, cmap="viridis", shading="nearest")
    for iface in stack.interfaces(x):
        ax.plot(x, iface, color="k", lw=0.8)
    ax.plot([70, 70], [z[-1], 1.0], color="w", lw=2)
    ax.scatter(px, pz, c="r", s=12, zorder=3)
    ax.set_title(f"realization {seed}")
    ax.set_xlabel("x / m")
axes[0].set_ylabel("z / m")
fig.tight_layout()
plt.show()

###############################################################################
# In contrast to the :any:`LayerStack` in the previous example, the
# contact elevations are honoured in all wells, also in ``W2``, which does
# not reach the base.
#
# In 3D, the interfaces of a stack are structured surfaces on the lateral
# grid and can be exported to `PyVista <https://docs.pyvista.org/>`__
# meshes with :any:`interfaces_to_pyvista`, e.g. for plotting or
# writing VTK files.

stack_3d = gs.SurfaceStack(
    [
        gs.Surface.from_model(gs.Matern(dim=2, var=6, len_scale=50), mean=m)
        for m in means
    ],
    dim=3,
    seed=19970221,
)
try:
    import pyvista as pv

    meshes = gs.geometry.interfaces_to_pyvista(stack_3d, (x[::4], x[::4]))
except ImportError:
    print("pyvista is not installed")
else:
    print([mesh.n_points for mesh in meshes])
    plotter = pv.Plotter()
    colors = ["saddlebrown", "goldenrod", "seagreen", "steelblue"]
    for i in range(len(meshes)):
        plotter.add_mesh(
            meshes[i], color=colors[i], opacity=0.8, label=names[i]
        )
    # plotter.add_legend()
    # plotter.show_grid()
    # plotter.show()

###############################################################################
# If you uncomment the three lines above, a plot like this will be shown.
#
# .. image:: ../../pics/geometry_surfaces_3d.png
#    :width: 400px
#    :align: center

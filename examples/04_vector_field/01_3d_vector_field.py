"""
Generating a Random 3D Vector Field
-----------------------------------

In this example we are going to generate a random 3D vector field with a
Gaussian covariance model. The mesh on which we generate the field will be
externally defined and it will be generated by PyVista.
"""

# sphinx_gallery_thumbnail_path = 'pics/GS_3d_vector_field.png'
import pyvista as pv

import gstools as gs

# mainly for setting a white background
pv.set_plot_theme("document")

###############################################################################
# create a uniform grid with PyVista
dims, spacing, origin = (40, 30, 10), (1, 1, 1), (-10, 0, 0)
mesh = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

###############################################################################
# create an incompressible random 3d velocity field on the given mesh
# with added mean velocity in x-direction
model = gs.Gaussian(dim=3, var=3, len_scale=1.5)
srf = gs.SRF(model, mean=(0.5, 0, 0), generator="VectorField", seed=198412031)
srf.mesh(mesh, points="points", name="Velocity")

###############################################################################
# Now, we can do the plotting
streamlines = mesh.streamlines(
    "Velocity",
    terminal_speed=0.0,
    n_points=800,
    source_radius=2.5,
)

# set a fancy camera position
cpos = [(25, 23, 17), (0, 10, 0), (0, 0, 1)]

p = pv.Plotter()
# adding an outline might help navigating in 3D space
# p.add_mesh(mesh.outline(), color="k")
p.add_mesh(
    streamlines.tube(radius=0.005),
    show_scalar_bar=False,
    diffuse=0.5,
    ambient=0.5,
)

###############################################################################
# .. note::
#    PyVista is not working on readthedocs, but you can try it out yourself by
#    uncommenting the following line of code.

# p.show(cpos=cpos)

###############################################################################
# The result should look like this:
#
# .. image:: ../../pics/GS_3d_vector_field.png
#    :width: 400px
#    :align: center

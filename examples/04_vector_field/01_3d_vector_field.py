"""
Generating a Random 3D Vector Field
-----------------------------------

In this example we are going to generate a random 3D vector field with a
Gaussian covariance model. The mesh on which we generate the field will be
externally defined and it will be generated by PyVista.
"""
import numpy as np
import gstools as gs
import pyvista as pv

pv.set_plot_theme("document")

###############################################################################
# create a uniform grid with PyVista
nx, ny, nz = 40, 30, 10
mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), (-10., 0., 0.))
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]

###############################################################################
# create an incompressible random 3d velocity field on the given mesh
model = gs.Gaussian(dim=3, var=3, len_scale=1.5)
srf = gs.SRF(model, generator='VectorField')
srf((x, y, z), mesh_type='unstructured', seed=198412031)

# add a mean velocity in x-direction
srf.field[0,:] += 0.5

###############################################################################
# add the velocity field to the mesh object
mesh["Velocity"] = srf.field.T

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
# .. image:: https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_3d_vector_field.png
#    :width: 400px
#    :align: center

"""
Using PyVista meshes
--------------------

`PyVista <https://www.pyvista.org>`__ is a helper module for the
Visualization Toolkit (VTK) that takes a different approach on interfacing with
VTK through NumPy and direct array access.

It provides mesh data structures and filtering methods for spatial datasets,
makes 3D plotting simple and is built for large/complex data geometries.

The :any:`Field.mesh` method enables easy field creation on PyVista meshes
used by the :any:`SRF` or :any:`Krige` class.
"""
# sphinx_gallery_thumbnail_path = 'https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_pyvista.png'
import pyvista as pv
import gstools as gs

###############################################################################
# We create a structured grid with PyVista containing of 50 segments
# on all three axis each with a length of 2 (whatever unit).

dim, spacing = (50, 50, 50), (2, 2, 2)
grid = pv.UniformGrid(dim, spacing)

###############################################################################
# Now we set up the SRF class as always. We'll use an anisotropic model.

model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=(0.8, 0.4, 0.2))
srf = gs.SRF(model, seed=19970221)

###############################################################################
# The PyVista mesh can now be directly passed to the :any:`SRF.mesh` method.
# When dealing with meshes, one can choose if the field should be generated
# on the mesh-points (`"points"`) of the cell-centroids (`"centroids"`).
#
# In addition we can set a name, under which the resulting field is stored
# in the mesh.

srf.mesh(grid, points="points", name="random-field")

###############################################################################
# Now we have access to PyVista's abundancy of methods to explore the field.
#
# Note
# ----
# PyVista is not working on readthedocs, but you can try it out yourself by
# uncommenting the following line of code.

# grid.contour(isosurfaces=8).plot()

###############################################################################
# The result should look like this:
#
# .. image:: https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_pyvista.png
#    :width: 400px
#    :align: center

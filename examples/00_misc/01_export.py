"""
Exporting Fields
----------------

GSTools provides simple exporting routines to convert generated fields to
`VTK <https://vtk.org/>`__ files.

These can be viewed for example with `Paraview <https://www.paraview.org/>`__.
"""

# sphinx_gallery_thumbnail_path = 'pics/paraview.png'
import gstools as gs

x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model)
field = srf((x, y), mesh_type="structured")
srf.vtk_export(filename="field")

###############################################################################
# The result displayed with Paraview:
#
# .. image:: ../../pics/paraview.png
#    :width: 400px
#    :align: center

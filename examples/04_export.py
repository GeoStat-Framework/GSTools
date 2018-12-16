from gstools import SRF, Gaussian, vtk_export
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
field = srf((x, y), mesh_type='structured')
vtk_export("field", (x, y), field, mesh_type='structured')

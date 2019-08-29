from gstools import SRF, Gaussian

x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
field = srf((x, y), mesh_type="structured")
srf.vtk_export("field")

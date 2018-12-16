from gstools import SRF, Gaussian
import matplotlib.pyplot as plt
# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
field = srf((x, y), mesh_type='structured')
plt.imshow(field)
plt.show()

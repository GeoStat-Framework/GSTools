"""
Combinations
------------

You can combine different transformations simply by successively applying them.

Here, we first force the single field realization to hold the given moments,
namely mean and variance.
Then we apply the Zinn & Harvey transformation to connect the low values.
Afterwards the field is transformed to a binary field and last but not least,
we transform it to log-values.

We can select the desired field by its name and we can define an output name
to store the field.

If you don't specify `field` and `store` everything happens inplace.
"""
# sphinx_gallery_thumbnail_number = 1
import gstools as gs

# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, mean=-9, seed=20170519)
srf.structured([x, y])
srf.transform("force_moments", field="field", store="f_forced")
srf.transform("zinnharvey", field="f_forced", store="f_zinnharvey", conn="low")
srf.transform("binary", field="f_zinnharvey", store="f_binary")
srf.transform("lognormal", field="f_binary", store="f_result")
srf.plot(field="f_result")

###############################################################################
# The resulting field could be interpreted as a transmissivity field, where
# the values of low permeability are the ones being the most connected
# and only two kinds of soil exist.
#
# All stored fields can be accessed and plotted by name:

print("Max binary value:", srf.f_binary.max())
srf.plot(field="f_zinnharvey")

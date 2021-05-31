"""
Standalone Field class
----------------------

The :any:`Field` class of GSTools can be used to plot arbitrary data in nD.

In the following example we will produce 10000 random points in 4D with
random values and plot them.
"""
import numpy as np
import gstools as gs

x1 = np.random.RandomState(19970221).rand(10000) * 100.0
x2 = np.random.RandomState(20011012).rand(10000) * 100.0
x3 = np.random.RandomState(20210530).rand(10000) * 100.0
x4 = np.random.RandomState(20210531).rand(10000) * 100.0
values = np.random.RandomState(2021).rand(10000) * 100.0

###############################################################################
# Only thing needed to instantiate the Field is the dimension.
#
# Afterwards we can call the instance like all other Fields
# (:any:`SRF`, :any:`Krige` or :any:`CondSRF`), but with an additional field.

plotter = gs.field.Field(dim=4)
plotter(pos=(x1, x2, x3, x4), field=values)
plotter.plot()

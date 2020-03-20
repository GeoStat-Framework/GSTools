==================
GSTools Quickstart
==================

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gstools.png
   :width: 150px
   :align: center

GeoStatTools provides geostatistical tools for random field generation and
variogram estimation based on many readily provided and even user-defined
covariance models.


Installation
============

conda
-----

GSTools can be installed via
`conda <https://docs.conda.io/en/latest/miniconda.html>`_ on Linux, Mac, and
Windows.
Install the package by typing the following command in a command terminal:

.. code-block:: none

    conda install gstools

In case conda forge is not set up for your system yet, see the easy to follow
instructions on `conda forge <https://github.com/conda-forge/gstools-feedstock#installing-gstools>`_.
Using conda, the parallelized version of GSTools should be installed.


pip
---

GSTools can be installed via `pip <https://pypi.org/project/gstools/>`_
on Linux, Mac, and Windows.
On Windows you can install `WinPython <https://winpython.github.io/>`_ to get
Python and pip running.
Install the package by typing the following into command in a command terminal:

.. code-block:: none

    pip install gstools

To get the latest development version you can install it directly from GitHub:

.. code-block:: none

    pip install git+git://github.com/GeoStat-Framework/GSTools.git@develop

If something went wrong during installation, try the :code:`-I` `flag from pip <https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i>`_.

To enable the OpenMP support, you have to provide a C compiler, Cython and OpenMP.
To get all other dependencies, it is recommended to first install gstools once
in the standard way just decribed.
Simply use the following commands:

.. code-block:: none

    pip install gstools
    pip install -I --no-deps --global-option="--openmp" gstools

Or for the development version:

.. code-block:: none

    pip install git+git://github.com/GeoStat-Framework/GSTools.git@develop
    pip install -I --no-deps --global-option="--openmp" git+git://github.com/GeoStat-Framework/GSTools.git@develop

The flags :code:`-I --no-deps` force pip to reinstall gstools but not the dependencies.


Citation
========

At the moment you can cite the Zenodo code publication of GSTools:

| *Sebastian Müller & Lennart Schüler. GeoStat-Framework/GSTools. Zenodo. https://doi.org/10.5281/zenodo.1313628*

If you want to cite a specific version, have a look at the Zenodo site.

A publication for the GeoStat-Framework is in preperation.


Tutorials and Examples
======================

The documentation also includes some `tutorials <tutorials.html>`__,
showing the most important use cases of GSTools, which are

- `Random Field Generation <examples/01_random_field/index.html>`__
- `The Covariance Model <examples/02_cov_model/index.html>`__
- `Variogram Estimation <examples/03_variogram/index.html>`__
- `Random Vector Field Generation <examples/04_vector_field/index.html>`__
- `Kriging <examples/05_kriging/index.html>`__
- `Conditioned random field generation <examples/06_conditioned_fields/index.html>`__
- `Field transformations <examples/07_transformations/index.html>`__
- `Miscellaneous examples <examples/00_misc/index.html>`__

Some more examples are provided in the examples folder.


Spatial Random Field Generation
===============================

The core of this library is the generation of spatial random fields.
These fields are generated using the randomisation method, described by
`Heße et al. 2014 <https://doi.org/10.1016/j.envsoft.2014.01.013>`_.


Examples
--------

Gaussian Covariance Model
^^^^^^^^^^^^^^^^^^^^^^^^^

This is an example of how to generate a 2 dimensional spatial random field (:any:`SRF`)
with a :any:`Gaussian` covariance model.

.. code-block:: python

    import gstools as gs
    # structured field with a size 100x100 and a grid-size of 1x1
    x = y = range(100)
    model = gs.Gaussian(dim=2, var=1, len_scale=10)
    srf = gs.SRF(model)
    srf((x, y), mesh_type='structured')
    srf.plot()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gau_field.png
   :width: 400px
   :align: center

A similar example but for a three dimensional field is exported to a
`VTK <https://vtk.org/>`__ file, which can be visualized with
`ParaView <https://www.paraview.org/>`_ or
`PyVista <https://docs.pyvista.org>`__ in Python:

.. code-block:: python

    import gstools as gs
    # structured field with a size 100x100x100 and a grid-size of 1x1x1
    x = y = z = range(100)
    model = gs.Gaussian(dim=3, var=0.6, len_scale=20)
    srf = gs.SRF(model)
    srf((x, y, z), mesh_type='structured')
    srf.vtk_export('3d_field') # Save to a VTK file for ParaView

    mesh = srf.to_pyvista() # Create a PyVista mesh for plotting in Python
    mesh.threshold_percent(0.5).plot()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/3d_gau_field.png
   :width: 400px
   :align: center


Estimating and fitting variograms
=================================

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.

All covariance models can be used to fit given variogram data by a simple interface.


Examples
--------

This is an example of how to estimate the variogram of a 2 dimensional unstructured field and estimate the parameters of the covariance
model again.

.. code-block:: python

    import numpy as np
    import gstools as gs
    # generate a synthetic field with an exponential model
    x = np.random.RandomState(19970221).rand(1000) * 100.
    y = np.random.RandomState(20011012).rand(1000) * 100.
    model = gs.Exponential(dim=2, var=2, len_scale=8)
    srf = gs.SRF(model, mean=0, seed=19970221)
    field = srf((x, y))
    # estimate the variogram of the field with 40 bins
    bins = np.arange(40)
    bin_center, gamma = gs.vario_estimate_unstructured((x, y), field, bins)
    # fit the variogram with a stable model. (no nugget fitted)
    fit_model = gs.Stable(dim=2)
    fit_model.fit_variogram(bin_center, gamma, nugget=False)
    # output
    ax = fit_model.plot(x_max=40)
    ax.plot(bin_center, gamma)
    print(fit_model)

Which gives:

.. code-block:: python

    Stable(dim=2, var=1.92, len_scale=8.15, nugget=0.0, anis=[1.], angles=[0.], alpha=1.05)

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/exp_vario_fit.png
   :width: 400px
   :align: center


Kriging and Conditioned Random Fields
=====================================

An important part of geostatistics is Kriging and conditioning spatial random
fields to measurements. With conditioned random fields, an ensemble of field realizations
with their variability depending on the proximity of the measurements can be generated.


Example
-------

For better visualization, we will condition a 1d field to a few "measurements",
generate 100 realizations and plot them:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import gstools as gs

    # conditions
    cond_pos = [0.3, 1.9, 1.1, 3.3, 4.7]
    cond_val = [0.47, 0.56, 0.74, 1.47, 1.74]

    gridx = np.linspace(0.0, 15.0, 151)

    # spatial random field class
    model = gs.Gaussian(dim=1, var=0.5, len_scale=2)
    srf = gs.SRF(model)
    srf.set_condition(cond_pos, cond_val, "ordinary")

    # generate the ensemble of field realizations
    fields = []
    for i in range(100):
        fields.append(srf(gridx, seed=i))
        plt.plot(gridx, fields[i], color="k", alpha=0.1)
    plt.scatter(cond_pos, cond_val, color="k")
    plt.show()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/cond_ens.png
   :width: 600px
   :align: center


User defined covariance models
==============================

One of the core-features of GSTools is the powerful
:any:`CovModel`
class, which allows to easy define covariance models by the user.


Example
-------

Here we re-implement the Gaussian covariance model by defining just the
`correlation <https://en.wikipedia.org/wiki/Autocovariance#Normalization>`_ function,
which takes a non-dimensional distance :class:`h = r/l`

.. code-block:: python

    import numpy as np
    import gstools as gs
    # use CovModel as the base-class
    class Gau(gs.CovModel):
        def cor(self, h):
            return np.exp(-h**2)

And that's it! With :class:`Gau` you now have a fully working covariance model,
which you could use for field generation or variogram fitting as shown above.


Incompressible Vector Field Generation
======================================

Using the original `Kraichnan method <https://doi.org/10.1063/1.1692799>`_, incompressible random
spatial vector fields can be generated.


Example
-------

.. code-block:: python

   import numpy as np
   import gstools as gs
   x = np.arange(100)
   y = np.arange(100)
   model = gs.Gaussian(dim=2, var=1, len_scale=10)
   srf = gs.SRF(model, generator='VectorField')
   srf((x, y), mesh_type='structured', seed=19841203)
   srf.plot()

yielding

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/vec_srf_tut_gau.png
   :width: 600px
   :align: center


VTK/PyVista Export
==================

After you have created a field, you may want to save it to file, so we provide
a handy `VTK <https://www.vtk.org/>`_ export routine using the :class:`.vtk_export()` or you could
create a VTK/PyVista dataset for use in Python with to :class:`.to_pyvista()` method:

.. code-block:: python

    import gstools as gs
    x = y = range(100)
    model = gs.Gaussian(dim=2, var=1, len_scale=10)
    srf = gs.SRF(model)
    srf((x, y), mesh_type='structured')
    srf.vtk_export("field") # Saves to a VTK file
    mesh = srf.to_pyvista() # Create a VTK/PyVista dataset in memory
    mesh.plot()

Which gives a RectilinearGrid VTK file :file:`field.vtr` or creates a PyVista mesh
in memory for immediate 3D plotting in Python.

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/pyvista_export.png
   :width: 600px
   :align: center


Requirements
============

- `Numpy >= 1.14.5 <http://www.numpy.org>`_
- `SciPy >= 1.1.0 <http://www.scipy.org>`_
- `hankel >= 1.0.2 <https://github.com/steven-murray/hankel>`_
- `emcee >= 3.0.0 <https://github.com/dfm/emcee>`_
- `pyevtk >= 1.1.1 <https://github.com/pyscience-projects/pyevtk>`_


Optional
--------

- `matplotlib <https://matplotlib.org>`_
- `pyvista <https://docs.pyvista.org>`_


License
=======

`LGPLv3 <https://github.com/GeoStat-Framework/GSTools/blob/master/LICENSE>`_

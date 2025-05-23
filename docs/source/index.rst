==================
GSTools Quickstart
==================

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/gstools.png
   :width: 150px
   :align: center

.. only:: html

   **Get in Touch!**

   |GH-Discussions| |Email|

   **Youtube Tutorial on GSTools**

   .. youtube:: qZBJ-AZXq6Q
      :width: 100%

   |

Purpose
=======

GSTools provides geostatistical tools for various purposes:

- random field generation, including periodic boundaries
- simple, ordinary, universal and external drift kriging
- conditioned field generation
- incompressible random vector field generation
- (automated) variogram estimation and fitting
- directional variogram estimation and modelling
- data normalization and transformation
- many readily provided and even user-defined covariance models
- metric spatio-temporal modelling
- plurigaussian field simulations (PGS)
- plotting and exporting routines


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

    pip install git+git://github.com/GeoStat-Framework/GSTools.git@main

If something went wrong during installation, try the :code:`-I` `flag from pip <https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i>`_.

**Speeding up GSTools by parallelization**

We provide two possibilities to run GSTools in parallel, often causing a
massive improvement in runtime. In either case, the number of parallel
threads can be set with the global variable `config.NUM_THREADS`. If not set,
all cores are used.
When using conda, the parallel version of GSTools is installed per default.

***Parallelizing Cython***

For parallel support, the `GSTools-Cython <https://github.com/GeoStat-Framework/GSTools-Cython>`_
backend needs to be compiled from source the following way:

.. code-block:: none

    export GSTOOLS_BUILD_PARALLEL=1
    pip install --no-binary=gstools-cython gstools

You have to provide a C compiler and OpenMP to compile GSTools-Cython with parallel support.
The feature is controlled by the environment variable
``GSTOOLS_BUILD_PARALLEL``, that can be ``0`` or ``1`` (interpreted as ``0`` if not present).
Note, that the ``--no-binary=gstools-cython`` option forces pip to not use a wheel
for the GSTools-Cython backend.

For the development version, you can do almost the same:

.. code-block:: none

    export GSTOOLS_BUILD_PARALLEL=1
    pip install git+git://github.com/GeoStat-Framework/GSTools-Cython.git@main
    pip install git+git://github.com/GeoStat-Framework/GSTools.git@main


***Using GSTools-Core for parallelization and even more speed***

You can install the optional dependency `GSTools-Core <https://github.com/GeoStat-Framework/GSTools-Core>`_,
which is a re-implementation of GSTools-Cython:

.. code-block:: none

    pip install gstools[rust]

or by manually installing the package

.. code-block:: none

    pip install gstools-core

The new package uses the language Rust and it should be safer and faster (in some cases by orders of magnitude).
Once the package GSTools-Core is available on your machine, it will be used by default.
In case you want to switch back to the Cython implementation, you can set
:code:`gstools.config.USE_GSTOOLS_CORE=False` in your code. This also works at runtime.

GSTools-Core will automatically run in parallel, without having to provide OpenMP or a local C compiler.


Citation
========

If you are using GSTools in your publication please cite our paper:

    Müller, S., Schüler, L., Zech, A., and Heße, F.: GSTools v1.3: a toolbox for geostatistical modelling in Python, Geosci. Model Dev., 15, 3161–3182, https://doi.org/10.5194/gmd-15-3161-2022, 2022.

You can cite the Zenodo code publication of GSTools by:

    Sebastian Müller & Lennart Schüler. GeoStat-Framework/GSTools. Zenodo. https://doi.org/10.5281/zenodo.1313628

If you want to cite a specific version, have a look at the `Zenodo site <https://doi.org/10.5281/zenodo.1313628>`__.


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
- `Geographic Coordinates <examples/08_geo_coordinates/index.html>`__
- `Spatio-Temporal Modelling <examples/09_spatio_temporal/index.html>`__
- `Normalizing Data <examples/10_normalizer/index.html>`__
- `Plurigaussian Field Generation (PGS) <examples/11_plurigaussian/index.html>`__
- `Miscellaneous examples <examples/00_misc/index.html>`__


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

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/gau_field.png
   :width: 400px
   :align: center

GSTools also provides support for `geographic coordinates <https://en.wikipedia.org/wiki/Geographic_coordinate_system>`_.
This works perfectly well with `cartopy <https://scitools.org.uk/cartopy/docs/latest/index.html>`_.

.. code-block:: python

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import gstools as gs
    # define a structured field by latitude and longitude
    lat = lon = range(-80, 81)
    model = gs.Gaussian(latlon=True, len_scale=777, geo_scale=gs.KM_SCALE)
    srf = gs.SRF(model, seed=12345)
    field = srf.structured((lat, lon))
    # Orthographic plotting with cartopy
    ax = plt.subplot(projection=ccrs.Orthographic(-45, 45))
    cont = ax.contourf(lon, lat, field, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    plt.colorbar(cont)

.. image:: https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_globe.png
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
    model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=(0.8, 0.4, 0.2))
    srf = gs.SRF(model)
    srf((x, y, z), mesh_type='structured')
    srf.vtk_export('3d_field') # Save to a VTK file for ParaView

    mesh = srf.to_pyvista() # Create a PyVista mesh for plotting in Python
    mesh.contour(isosurfaces=8).plot()

.. image:: https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_pyvista.png
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
    # estimate the variogram of the field
    bin_center, gamma = gs.vario_estimate((x, y), field)
    # fit the variogram with a stable model. (no nugget fitted)
    fit_model = gs.Stable(dim=2)
    fit_model.fit_variogram(bin_center, gamma, nugget=False)
    # output
    ax = fit_model.plot(x_max=max(bin_center))
    ax.scatter(bin_center, gamma)
    print(fit_model)

Which gives:

.. code-block:: python

    Stable(dim=2, var=1.85, len_scale=7.42, nugget=0.0, anis=[1.0], angles=[0.0], alpha=1.09)

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GeoStat-Framework.github.io/master/img/GS_vario_est.png
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

    # conditioned spatial random field class
    model = gs.Gaussian(dim=1, var=0.5, len_scale=2)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)
    cond_srf = gs.CondSRF(krige)
    # same output positions for all ensemble members
    grid_pos = np.linspace(0.0, 15.0, 151)
    cond_srf.set_pos(grid_pos)

    # seeded ensemble generation
    seed = gs.random.MasterRNG(20170519)
    for i in range(100):
        field = cond_srf(seed=seed(), store=f"field_{i}")
        plt.plot(grid_pos, field, color="k", alpha=0.1)
    plt.scatter(cond_pos, cond_val, color="k")
    plt.show()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/cond_ens.png
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
   srf = gs.SRF(model, generator='VectorField', seed=19841203)
   srf((x, y), mesh_type='structured')
   srf.plot()

yielding

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/vec_srf_tut_gau.png
   :width: 600px
   :align: center


Plurigaussian Field Simulation (PGS)
====================================

With PGS, more complex categorical (or discrete) fields can be created.


Example
-------

.. code-block:: python

   import gstools as gs
   import numpy as np
   import matplotlib.pyplot as plt

   N = [180, 140]

   x, y = range(N[0]), range(N[1])

   # we need 2 SRFs
   model = gs.Gaussian(dim=2, var=1, len_scale=5)
   srf = gs.SRF(model)
   field1 = srf.structured([x, y], seed=20170519)
   field2 = srf.structured([x, y], seed=19970221)

   # with `lithotypes`, we prescribe the categorical data and its relations
   # here, we use 2 categories separated by a rectangle.
   rect = [40, 32]
   lithotypes = np.zeros(N)
   lithotypes[
       N[0] // 2 - rect[0] // 2 : N[0] // 2 + rect[0] // 2,
       N[1] // 2 - rect[1] // 2 : N[1] // 2 + rect[1] // 2,
   ] = 1

   pgs = gs.PGS(2, [field1, field2])
   P = pgs(lithotypes)

   fig, axs = plt.subplots(1, 2)
   axs[0].imshow(lithotypes, cmap="copper")
   axs[1].imshow(P, cmap="copper")
   plt.show()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/2d_pgs.png
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

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/pyvista_export.png
   :width: 600px
   :align: center


Requirements
============

- `NumPy >= 1.20.0 <http://www.numpy.org>`_
- `SciPy >= 1.1.0 <http://www.scipy.org>`_
- `hankel >= 1.0.0 <https://github.com/steven-murray/hankel>`_
- `emcee >= 3.0.0 <https://github.com/dfm/emcee>`_
- `pyevtk >= 1.1.1 <https://github.com/pyscience-projects/pyevtk>`_
- `meshio >= 5.1.0 <https://github.com/nschloe/meshio>`_


Optional
--------

- `GSTools-Core >= 0.2.0 <https://github.com/GeoStat-Framework/GSTools-Core>`_
- `matplotlib <https://matplotlib.org>`_
- `pyvista <https://docs.pyvista.org>`_


Contact
-------

You can contact us via `info@geostat-framework.org <mailto:info@geostat-framework.org>`_.


License
=======

`LGPLv3 <https://github.com/GeoStat-Framework/GSTools/blob/main/LICENSE>`_


.. |GH-Discussions| image:: https://img.shields.io/badge/GitHub-Discussions-f6f8fa?logo=github&style=flat
    :alt: GH-Discussions
    :target: https://github.com/GeoStat-Framework/GSTools/discussions
.. |Email| image:: https://img.shields.io/badge/Email-GeoStat--Framework-468a88?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbDpzcGFjZT0icHJlc2VydmUiIHdpZHRoPSI1MDAiIGhlaWdodD0iNTAwIj48cGF0aCBkPSJNNDQ4IDg4SDUyYy0yNyAwLTQ5IDIyLTQ5IDQ5djIyNmMwIDI3IDIyIDQ5IDQ5IDQ5aDM5NmMyNyAwIDQ5LTIyIDQ5LTQ5VjEzN2MwLTI3LTIyLTQ5LTQ5LTQ5em0xNiA0OXYyMjZsLTIgNy0xMTUtMTE2IDExNy0xMTd6TTM2IDM2M1YxMzdsMTE3IDExN0wzOCAzNzBsLTItN3ptMjE5LTYzYy0zIDMtNyAzLTEwIDBMNjYgMTIxaDM2OHptLTc5LTIzIDQ2IDQ2YTM5IDM5IDAgMCAwIDU2IDBsNDYtNDYgMTAxIDEwMkg3NXoiIHN0eWxlPSJmaWxsOiNmNWY1ZjU7ZmlsbC1vcGFjaXR5OjEiLz48L3N2Zz4=
    :alt: Email
    :target: mailto:info@geostat-framework.org

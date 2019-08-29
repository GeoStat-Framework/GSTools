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

The package can be installed via `pip <https://pypi.org/project/gstools/>`_.
On Windows you can install `WinPython <https://winpython.github.io/>`_ to get
Python and pip running. Also `conda provides pip support <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`_.
Install GSTools by typing the following into the command prompt:

.. code-block:: none

    pip install gstools

To get the latest development version you can install it directly from GitHub:

.. code-block:: none

    pip install https://github.com/GeoStat-Framework/GSTools/archive/master.zip

To enable the OpenMP support, you have to provide a C compiler, Cython and OpenMP.
To get all other dependencies, it is recommended to first install gstools once
in the standard way just decribed.
Then use the following command:

.. code-block:: none

    pip install --global-option="--openmp" gstools

Or for the development version:

.. code-block:: none

    pip install --global-option="--openmp" https://github.com/GeoStat-Framework/GSTools/archive/master.zip

If something went wrong during installation, try the :code:`-I` `flag from pip <https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i>`_.


Citation
========

At the moment you can cite the Zenodo code publication of GSTools:

| *Sebastian Müller, & Lennart Schüler. (2019, January 18). GeoStat-Framework/GSTools: Bouncy Blue (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.2543658*

A publication for the GeoStat-Framework is in preperation.

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

    from gstools import SRF, Gaussian
    import matplotlib.pyplot as plt
    # structured field with a size 100x100 and a grid-size of 1x1
    x = y = range(100)
    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model)
    srf((x, y), mesh_type='structured')
    srf.plot()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gau_field.png
   :width: 400px
   :align: center

A similar example but for a three dimensional field is exported to a
`VTK <https://vtk.org/>`__ file, which can be visualized with
`ParaView <https://www.paraview.org/>`_.

.. code-block:: python

    from gstools import SRF, Gaussian
    import matplotlib.pyplot as pt
    # structured field with a size 100x100x100 and a grid-size of 1x1x1
    x = y = z = range(100)
    model = Gaussian(dim=3, var=0.6, len_scale=20)
    srf = SRF(model)
    srf((x, y, z), mesh_type='structured')
    srf.vtk_export('3d_field')

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/3d_gau_field.png
   :width: 400px
   :align: center


Truncated Power Law Model
^^^^^^^^^^^^^^^^^^^^^^^^^

GSTools also implements truncated power law variograms, which can be represented as a
superposition of scale dependant modes in form of standard variograms, which are truncated by
a lower- :math:`\ell_{\mathrm{low}}` and an upper length-scale :math:`\ell_{\mathrm{up}}`.

This example shows the truncated power law (:any:`TPLStable`) based on the
:any:`Stable` covariance model and is given by

.. math::
   \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) =
   \intop_{\ell_{\mathrm{low}}}^{\ell_{\mathrm{up}}}
   \gamma(r,\lambda) \frac{\rm d \lambda}{\lambda}

with `Stable` modes on each scale:

.. math::
   \gamma(r,\lambda) &=
   \sigma^2(\lambda)\cdot\left(1-
   \exp\left[- \left(\frac{r}{\lambda}\right)^{\alpha}\right]
   \right)\\
   \sigma^2(\lambda) &= C\cdot\lambda^{2H}

which gives Gaussian modes for ``alpha=2`` or Exponential modes for ``alpha=1``.

For :math:`\ell_{\mathrm{low}}=0` this results in:

.. math::
   \gamma_{\ell_{\mathrm{up}}}(r) &=
   \sigma^2_{\ell_{\mathrm{up}}}\cdot\left(1-
   \frac{2H}{\alpha} \cdot
   E_{1+\frac{2H}{\alpha}}
   \left[\left(\frac{r}{\ell_{\mathrm{up}}}\right)^{\alpha}\right]
   \right) \\
   \sigma^2_{\ell_{\mathrm{up}}} &=
   C\cdot\frac{\ell_{\mathrm{up}}^{2H}}{2H}

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gstools import SRF, TPLStable
    x = y = np.linspace(0, 100, 100)
    model = TPLStable(
        dim=2,           # spatial dimension
        var=1,           # variance (C calculated internally, so that `var` is 1)
        len_low=0,       # lower truncation of the power law
        len_scale=10,    # length scale (a.k.a. range), len_up = len_low + len_scale
        nugget=0.1,      # nugget
        anis=0.5,        # anisotropy between main direction and transversal ones
        angles=np.pi/4,  # rotation angles
        alpha=1.5,       # shape parameter from the stable model
        hurst=0.7,       # hurst coefficient from the power law
    )
    srf = SRF(model, mean=1, mode_no=1000, seed=19970221, verbose=True)
    srf((x, y), mesh_type='structured')
    srf.plot()

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/tplstable_field.png
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
    from gstools import SRF, Exponential, Stable, vario_estimate_unstructured
    # generate a synthetic field with an exponential model
    x = np.random.RandomState(19970221).rand(1000) * 100.
    y = np.random.RandomState(20011012).rand(1000) * 100.
    model = Exponential(dim=2, var=2, len_scale=8)
    srf = SRF(model, mean=0, seed=19970221)
    field = srf((x, y))
    # estimate the variogram of the field with 40 bins
    bins = np.arange(40)
    bin_center, gamma = vario_estimate_unstructured((x, y), field, bins)
    # fit the variogram with a stable model. (no nugget fitted)
    fit_model = Stable(dim=2)
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


User defined covariance models
==============================

One of the core-features of GSTools is the powerfull
:any:`CovModel`
class, which allows to easy define covariance models by the user.


Example
-------

Here we re-implement the Gaussian covariance model by defining just the
`correlation <https://en.wikipedia.org/wiki/Autocovariance#Normalization>`_ function,
which takes a non-dimensional distance :class:`h = r/l`

.. code-block:: python

    from gstools import CovModel
    import numpy as np
    # use CovModel as the base-class
    class Gau(CovModel):
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
   import matplotlib.pyplot as plt
   from gstools import SRF, Gaussian
   x = np.arange(100)
   y = np.arange(100)
   model = Gaussian(dim=2, var=1, len_scale=10)
   srf = SRF(model, generator='VectorField')
   srf((x, y), mesh_type='structured', seed=19841203)
   srf.plot()

yielding

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/vec_srf_tut_gau.png
   :width: 600px
   :align: center


VTK Export
==========

After you have created a field, you may want to save it to file, so we provide
a handy `VTK <https://www.vtk.org/>`__ export routine:

.. code-block:: python

    from gstools import SRF, Gaussian
    x = y = range(100)
    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model)
    srf((x, y), mesh_type='structured')
    srf.vtk_export("field")

Which gives a RectilinearGrid VTK file ``field.vtr``.


Requirements
============

- `Numpy >= 1.14.5 <http://www.numpy.org>`_
- `SciPy >= 1.1.0 <http://www.scipy.org>`_
- `hankel >= 0.3.6 <https://github.com/steven-murray/hankel>`_
- `emcee <https://github.com/dfm/emcee>`_
- `pyevtk <https://bitbucket.org/pauloh/pyevtk>`_
- `six <https://github.com/benjaminp/six>`_


License
=======

`LGPLv3 <https://github.com/GeoStat-Framework/GSTools/blob/master/LICENSE>`_ © 2018-2019

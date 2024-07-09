Random Field Generation
=======================

The main feature of GSTools is the spatial random field generator :any:`SRF`,
which can generate random fields following a given covariance model.
The generator provides a lot of nice features, which will be explained in
the following

GSTools generates spatial random fields with a given covariance model or
semi-variogram. This is done by using the so-called randomization method.
The spatial random field is represented by a stochastic Fourier integral
and its discretised modes are evaluated at random frequencies.

In case you want to generate spatial random fields with periodic boundaries,
you can use the so-called Fourier method. See the corresponding examples for
how to do that. The spatial random field is represented by a stochastic
Fourier integral and its discretised modes are evaluated at equidistant
frequencies.

GSTools supports arbitrary and non-isotropic covariance models.

Examples
--------

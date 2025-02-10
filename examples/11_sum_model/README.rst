Summing Covariance Models
=========================

In geostatistics, the spatial variability of natural phenomena is often represented using covariance models,
which describe how values of a property correlate over distance.
A single covariance model may capture specific features of spatial correlation, such as smoothness or the range of influence.
However, many real-world spatial processes are complex, involving multiple overlapping structures
that cannot be adequately described by a single covariance model.

This is where **sum models** come into play.
A sum model combines multiple covariance models into a single representation,
allowing for a more flexible and comprehensive description of spatial variability.
By summing covariance models, we can:

1. **Capture Multi-Scale Variability:** Many spatial phenomena exhibit variability at different scales.
   For example, soil properties may have small-scale variation due to local heterogeneities and large-scale variation due to regional trends.
   A sum model can combine short-range and long-range covariance models to reflect this behavior.
2. **Incorporate Multiple Physical Processes:** Different processes may contribute to the observed spatial pattern.
   For instance, in hydrology, the spatial distribution of groundwater levels might be influenced by both geological structures and human activities.
   A sum model can represent these independent contributions.
3. **Improve Model Fit and Prediction Accuracy:** By combining models, sum models can better match empirical variograms or other observed data,
   leading to more accurate predictions in kriging or simulation tasks.
4. **Enhance Interpretability:** Each component of a sum model can be associated with a specific spatial process or scale,
   providing insights into the underlying mechanisms driving spatial variability.

The new :any:`SumModel` introduced in GSTools makes it straightforward to define and work with such composite covariance structures.
It allows users to combine any number of base models, each with its own parameters, in a way that is both intuitive and computationally efficient.

In the following tutorials, we'll explore how to use the :any:`SumModel` in GSTools,
including practical examples that demonstrate its utility in real-world scenarios.

Examples
--------

## Outline ##

The scripts folder contains scripts that use the ionotomo package for specific reasons. 
Ideally, one script tries to answer a question, or provide part of an answer to a question.
Thus, I state the questions that are to be addressed.

1. In `2d_vs_3d`: In terms of sufficiently calibrating the propagation effects of the ionosphere, is 2D enough, or is 3D required?

2. In `rvw_data_analysis`: Import Reinouts data and make bayesian analysis to infer some temporal and spatial properties.

3. In `3d_model_validation`: show the inversion pipeline works for simulated data, and (eventually) on real data (likely Reinout's).


## Simulating a simple ionosphere ##

Very simply one wishes to be able to provide a basic model for the ionosphere.
In this case the model has a bulk strucutre following the Internation Refernce Ionosphere (IRI) for the electron density.
The electron density thus defines, at a specific frequency, the refractive index at all points in the field.
I then impose a turbulent structure on this bulk layered structure following inspiration of Kolmogorov turbulence. 

Such turbulence supposes that the phase distortions in a thin layer are zero mean, with a spatial coherence (aka structure function) that follows a specfic dependence between an inner and outer scale. 
The structure function is thought to scale as $r^{2/3}$.
We approximate this with nu=2/3 Matern isotropic covariance.

The perturbations are in log-electron density space.

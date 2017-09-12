## Outline ##

The scripts folder contains scripts that use the ionotomo package for specific reasons. 
Ideally, one script tries to answer a question, or provide part of an answer to a question.
Thus, I state the questions that are to be addressed.

## Simulating a simple ionosphere ##

Very simply one wishes to be able to provide a basic model for the ionosphere.
I'll take this as the a priori model in the following so it should be not too simple.
In this case the model has a bulk strucutre following a parametrised fit of Chapman layers for the electron density.
The electron density thus defines, at a specific frequency, the refractive index at all points in the field.
I then impose a turbulent structure on this bulk layered structure following inspiration of Kolmogorov turbulence. 
Such turbulence supposes that the phase distortions in a thin layer are zero mean, with a spatial coherence (aka structure function) that follows a specfic dependence between an inner and outer scale. 
The structure function is thought to scale as $r^{2/3}$.

Taking the Chapman layers as the mean, I then expand around this point the refractive index to first order $n \approx n_0 - K*(ne - ne_0)$.
Thus, I take $n - n_0 \sim N(0, C_n)$ which provides a measure of $ne - ne_0$.

The magnatude of the fluctuations $C_n$

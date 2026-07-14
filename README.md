# *Nefes* — network solver for reacting compressible flows and thermoacoustics

Nefes models a fluid system as a directed graph of lumped elements and solves the steady mean flow and the linear perturbations around it — acoustics, entropy waves, and compositional disturbances — without resolving the full three-dimensional field.
The mean flow and its perturbations share one assembled operator: the converged Jacobian is also the zero-frequency perturbation network, this ensures the mean flow and the perturbation analysis are consistent by design.

## Capabilities

- Reacting compressible mean flows in the subsonic regime, including choking at a sonic throat
- Chemical-equilibrium thermochemistry (NASA–Glenn/CEA species data)
- Entropy and compositional (indirect) noise
- Linear stability analysis: eigenmode search (Beyn contour-integral method) and real-frequency Nyquist criterion
- Forced-response analysis
- Scattering and transfer matrices
- Identification of an unknown element's dynamic response (e.g. a flame) given a model of the rest

## Installation

## Documentation

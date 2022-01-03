# Re-envisioning NIFTy

## JAX

The (soft linked) code in this directory is a new interface for NIFTy written in JAX.
Some features of this new API are straight-forward re-implementations of features in NIFTy while other features are orthogonal to NIFTy and follow a different, usually more functional approach.
All essential pieces of NIFTy are implemented and the API is capable of (almost) fully replacing NIFTy's current NumPy based implementation.

### Current Features

* MAP
* MGVI
* geoVI
* Non-parametric correlated field

### TODO

The likelihood (or the Hamiltonian) probably is the object where it makes the most sense to translate to a different interface.
The minimization can be different depending on the API used but the likelihood should be a common denominator.
Inference schemes like MGVI, geoVI or MAP do not need to be similar nor should they be.
For all of these methods a more functional approach is desired instead.

Overall, it would make sense to re-implement `optimize_kl` from NIFTy because it abstracts away many details of how MGVI, geoVI or MAP is implemented.
Furthermore, this would make transitioning from NumPy NIFTy to a JAX-based NIFTy more easy while at the same time allowing for many changes to the interfaces of MGVI, geoVI and MAP.

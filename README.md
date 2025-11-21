# cfenv

## Building Python software environments for the Cosmo Foundry.

This package provides a Python conda environment with enough I/O and
AI/ML packages to enable users to get started exploring cosmology
datasets for AI models.

It is not intended to be the single environment that will meet the needs
of all users. More expert users can use the recipes in `scripts/` to
build their own environments, adding additional packages that they need
and/or dropping packages that they don't need to minimize the size and
complexity of the environment.

For starters this is focused on NERSC, but could be expanded for elsewhere.
The default environment provides:

  * **Core**: numpy scipy astropy pandas numba cupy mpi4py
  * **I/O**: pyarrow fastparquet fitsio h5py lsdb
  * **AI/ML**: pytorch jax sklearn
  * **Utilities**: ipython matplotlib pytest healpy requests
  * **Dataset APIs**: polymathic-aion, datasets, MultimodalUniverse, desispec

## Testing the environment

After installing and activating an environment, you can run a basic set of
checks with `pytest` in the top-level directory of cfenv.  This will check
that e.g. pytorch was installed with GPU support.

## Caveats

The NERSC environment currently does *not* include tensorflow due to
installation issues. To use tensorflow in alternate environments, see
https://docs.nersc.gov/machinelearning/tensorflow/ .

HuggingFace datasets/4.x is not compatible with MultimodalUniverse/v1.
Downgrading to datasets==3.6.0 would support MMU/v1, but that would require
downgrading numpy<2.x due to incompatibility in datasets.  In the meantime,
this environment has datasets/4.x, numpy/2.x, and MMU can be used by directly
reading files rather than through the HuggingFace datasets interface.
For further details see MultimodelUniverse
[#178](https://github.com/MultimodalUniverse/MultimodalUniverse/issues/178)
and
[#179](https://github.com/MultimodalUniverse/MultimodalUniverse/pull/179).


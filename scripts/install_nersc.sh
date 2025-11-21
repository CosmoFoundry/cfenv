#!/bin/bash

# -----
# Install a new CosmoFoundry environment at NERSC
# -----

# One and only one arg: the directory for the installation
if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $(basename "$0") INSTALL_DIR" >&2
    exit 1
fi

# immediately exit upon error
set -e

# pre-flight announcements
INSTALL_DIR="$1"
echo --- Installing new environment to $INSTALL_DIR at $(date)

# bash-fu: force all subsequent commands to log to output logfile
LOGDIR=$INSTALL_DIR/log
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/install.log
exec 3> $LOGFILE                  # fd 3 = log file (append)
# Fork a tee that writes both to the terminal (fd 1) and the log.
exec > >(tee -a $LOGFILE)         # stdout -> tee -> log
exec 2>&1                         # stderr -> same place as stdout

# Copy this script to the log directory for the record
cp -p ${BASH_SOURCE[0]} $LOGDIR

# -----
# This is where the real installation work begins
# -----

# Load NERSC environment modules
module load conda
module load cudatoolkit/12.9
module load gcc-native/13.2

# Create conda environment with a bunch of packages
echo --- Creating conda environment at $(date)
conda create --yes --prefix $INSTALL_DIR \
    numpy scipy astropy pandas numba cupy \
    pyarrow fastparquet fitsio h5py lsdb \
    jax scikit-learn \
    ipython matplotlib pytest healpy requests

conda activate $INSTALL_DIR

# PyTorch
echo --- Installing pytorch at $(date)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# TODO: tensorflow

# mpi4py requires special options to be compatible with CRAY mpich
echo --- Installing mpi4py at $(date)
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

# Install additional packages via pip
echo --- Adding additional packages from pip at $(date)
pip install polymathic-aion
pip install datasets
pip install git+https://github.com/MultimodalUniverse/MultimodalUniverse.git

# Dataset specific: DESI
echo --- Installing DESI utility code at $(date)
pip install desiutil
pip install git+https://github.com/desihub/desitarget
pip install --no-build-isolation git+https://github.com/desihub/desispec
pip install git+https://github.com/desihub/redrock
install_redrock_templates

# Generate a setup script to activate this environment, including the NERSC modules
cat > $INSTALL_DIR/setup.sh <<EOF
# Load required NERSC modules
module load conda
module load cudatoolkit/12.9
module load gcc-native/13.2

# Activate environment
conda activate ${INSTALL_DIR}

# For convenience
export COSMODATA=/dvs_ro/cfs/cdirs/cosmo/data
EOF

# all done
echo --- Done creating software environment at $(date)
echo --- To activate, please run
echo
echo "source $INSTALL_DIR/setup.sh"

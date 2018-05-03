FROM debian:testing-slim

RUN apt-get update && apt-get install -y \
    # Needed for gitlab tests
    git \
    # Packages needed for NIFTy
    libfftw3-dev \
    python python-pip python-dev python-future python-scipy \
    python3 python3-pip python3-dev python3-future python3-scipy \
    # Documentation build dependencies
    python-sphinx python-sphinx-rtd-theme python-numpydoc \
    # Testing dependencies
    python-nose python-parameterized \
    python3-nose python3-parameterized \
    # Optional NIFTy dependencies
    openmpi-bin libopenmpi-dev python-mpi4py python3-mpi4py \
    # Packages needed for NIFTy
  && pip install pyfftw \
  && pip3 install pyfftw \
  # Optional NIFTy dependencies
  && pip install git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git \
  # Testing dependencies
  && pip install coverage \
  && rm -rf /var/lib/apt/lists/*

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Packages needed for NIFTy
    python3-scipy \
    # Documentation build dependencies
    dvipng texlive-latex-base texlive-latex-extra \
    # Testing dependencies
    python3-pytest-cov jupyter \
    # Optional NIFTy dependencies
    python3-mpi4py python3-matplotlib python3-h5py python3-astropy \
  # more optional NIFTy dependencies
  && DUCC0_OPTIMIZATION=portable pip3 install ducc0 finufft jupyter jax jaxlib sphinx pydata-sphinx-theme jupytext \
  && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend
ENV MPLBACKEND agg

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

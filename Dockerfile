FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Documentation build dependencies
    dvipng texlive-latex-base texlive-latex-extra \
    # Dependency of mpi4py
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN DUCC0_OPTIMIZATION=portable pip3 install \
    # Packages needed for NIFTy
    scipy \
    # Optional nifty dependencies
    matplotlib h5py astropy ducc0 jax jaxlib mpi4py \
    # Testing dependencies
    pytest pytest-cov \
    # Documentation build dependencies
    jupyter nbconvert jupytext sphinx pydata-sphinx-theme

# Set matplotlib backend
ENV MPLBACKEND agg

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

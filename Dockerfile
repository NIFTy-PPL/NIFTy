FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Documentation build dependencies
    dvipng texlive-latex-base texlive-latex-extra pandoc \
    # Dependency of mpi4py
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages \
    # Packages needed for NIFTy
    scipy \
    # Optional nifty dependencies
    matplotlib h5py astropy ducc0 "jax!=0.6.0" "jaxlib!=0.6.0" jaxbind mpi4py healpy \
    # Testing dependencies
    pytest pytest-cov pytest-xdist \
    # Documentation build dependencies
    jupyter nbconvert jupytext "sphinx<8.2" pydata-sphinx-theme myst-parser sphinxcontrib-bibtex

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash runner
USER runner
WORKDIR /home/runner

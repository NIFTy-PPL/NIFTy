FROM debian:testing-slim

RUN apt-get update && apt-get install -y \
    # Needed for gitlab tests
    git \
    # Packages needed for NIFTy
    libfftw3-dev \
    python3 python3-pip python3-dev python3-scipy \
    # Documentation build dependencies
    python3-sphinx python3-sphinx-rtd-theme \
    # Testing dependencies
    python3-coverage python3-pytest python3-pytest-cov \
    # Optional NIFTy dependencies
    openmpi-bin libopenmpi-dev python3-mpi4py \
  # Optional NIFTy dependencies
  && pip3 install git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git \
  # Testing dependencies
  && rm -rf /var/lib/apt/lists/*

# Needed for demos to be running
RUN apt-get update && apt-get install -y python3-matplotlib \
  && python3 -m pip install --upgrade pip && python3 -m pip install jupyter \
  && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend
ENV MPLBACKEND agg

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

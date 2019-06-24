FROM debian:testing-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Packages needed for NIFTy
    python3-scipy \
    # Documentation build dependencies
    python3-sphinx-rtd-theme dvipng texlive-latex-base texlive-latex-extra \
    # Testing dependencies
    python3-pytest-cov jupyter \
    # Optional NIFTy dependencies
    python3-mpi4py python3-matplotlib \
  # more optional NIFTy dependencies
  && pip3 install git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git@master \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git \
  && pip3 install jupyter \
  && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend
ENV MPLBACKEND agg

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

FROM debian:testing-slim

RUN apt-get update

# Needed for gitlab tests
RUN apt-get install -y git

# Packages needed for NIFTy
RUN apt-get install -y libfftw3-dev
RUN apt-get install -y python python-dev python-future python-scipy
RUN apt-get install -y python3 python3-dev python3-future python3-scipy

# MPI requirements (optional NIFTy dependencies)
RUN apt-get install -y openmpi-bin libopenmpi-dev python-mpi4py python3-mpi4py

# Documentation build dependencies
RUN apt-get install -y python-sphinx python-sphinx-rtd-theme python-numpydoc

# Testing dependencies
RUN apt-get install -y python-nose 
RUN apt-get install -y python3-nose 

# Python module installations
RUN apt-get install -y python-pip python3-pip
RUN pip install pyfftw git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
RUN pip3 install pyfftw git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

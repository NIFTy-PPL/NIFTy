FROM debian:testing-slim

RUN apt-get update

# Debian package installations
# Packages needed for NIFTy
RUN apt-get install -y git libfftw3-dev openmpi-bin libopenmpi-dev
RUN apt-get install -y python python-pip python-dev python-matplotlib python-future python-mpi4py python-scipy
RUN apt-get install -y python3 python3-pip python3-dev python3-matplotlib python3-future python3-mpi4py python3-scipy

# Packages needed for generating the documentation
RUN apt-get install -y python-sphinx python-sphinx-rtd-theme python-numpydoc

# Packages needed for running tests
RUN apt-get install -y python-nose python-parameterized 
RUN apt-get install -y python3-nose python3-parameterized

# Python module installations
RUN pip install coverage pyfftw git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
RUN pip3 install pyfftw git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

FROM ubuntu:latest

# dependencies via apt
RUN apt-get update 
ADD ci/install_basics.sh /tmp/install_basics.sh
RUN cd /tmp && chmod +x install_basics.sh && ./install_basics.sh


# python dependencies
ADD ci/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt

ADD ci/requirements_extras.txt /tmp/requirements_extras.txt
RUN pip install --upgrade -r /tmp/requirements_extras.txt


# install pyHealpix, pyfftw and h5py
ADD ci/install_pyHealpix.sh /tmp/install_pyHealpix.sh
RUN cd /tmp && chmod +x install_pyHealpix.sh && ./install_pyHealpix.sh

ADD ci/install_mpi4py.sh /tmp/install_mpi4py.sh
RUN cd /tmp && chmod +x install_mpi4py.sh && ./install_mpi4py.sh

ADD ci/install_pyfftw.sh /tmp/install_pyfftw.sh
RUN cd /tmp && chmod +x install_pyfftw.sh && ./install_pyfftw.sh

ADD ci/install_h5py.sh /tmp/install_h5py.sh
RUN cd /tmp && chmod +x install_h5py.sh && ./install_h5py.sh


# copy sources and install nifty
COPY . /tmp/NIFTy
RUN pip install /tmp/NIFTy


# Cleanup
RUN rm -r /tmp/*

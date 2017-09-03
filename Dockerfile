FROM ubuntu:latest

# dependencies via apt
RUN apt-get update
ADD ci/install_basics.sh /tmp/install_basics.sh
RUN cd /tmp && chmod +x install_basics.sh && ./install_basics.sh


# python dependencies
ADD ci/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt


# install pyHealpix, pyfftw and h5py
ADD ci/install_pyHealpix.sh /tmp/install_pyHealpix.sh
RUN cd /tmp && chmod +x install_pyHealpix.sh && ./install_pyHealpix.sh


# copy sources and install nifty
COPY . /tmp/NIFTy
RUN pip install /tmp/NIFTy


# Cleanup
RUN rm -r /tmp/*

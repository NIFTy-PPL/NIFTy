#!/bin/bash

apt-get install -y libhdf5-10 libhdf5-dev libhdf5-openmpi-10 libhdf5-openmpi-dev hdf5-tools
CC=mpicc HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi HDF5_MPI="ON" pip install --no-binary=h5py h5py

#!/bin/bash

export CC=mpicc
export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
export HDF5_MPI="ON"
pip install --no-binary=h5py h5py

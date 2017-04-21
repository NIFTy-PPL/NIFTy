#!/bin/bash

wget https://api.github.com/repos/h5py/h5py/tags -O - | grep tarball_url | grep -v rc | head -n 1 | cut -d '"' -f 4 | wget -i - -O h5py.tar.gz
tar xzf h5py.tar.gz
cd h5py-h5py*
export CC=mpicc
export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
python setup.py configure --mpi
python setup.py build
python setup.py install
cd ..
rm -r h5py-h5py*
rm h5py.tar.gz

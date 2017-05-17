#!/bin/bash

apt-get install -y libatlas-base-dev libfftw3-bin libfftw3-dev libfftw3-double3 libfftw3-long3 libfftw3-mpi-dev libfftw3-mpi3 libfftw3-quad3 libfftw3-single3

git clone -b mpi https://github.com/fredros/pyFFTW.git
cd pyFFTW/
CC=mpicc python setup.py build_ext install
cd ..
rm -r pyFFTW

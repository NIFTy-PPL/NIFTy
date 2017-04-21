#!/bin/bash

git clone -b mpi https://github.com/fredRos/pyFFTW.git
cd pyFFTW/
CC=mpicc python setup.py build_ext install
cd ..
rm -r pyFFTW

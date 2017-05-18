#!/bin/bash

git clone https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
cd pyHealpix
autoreconf -i && ./configure --enable-openmp --enable-native-optimizations && make -j4 install
cd ..
rm -rf pyHealpix

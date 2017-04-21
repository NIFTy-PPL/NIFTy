#!/bin/bash

git clone https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
cd pyHealpix
autoreconf -i && ./configure && make -j4 install
cd ..
rm -rf pyHealpix

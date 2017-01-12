#!/bin/bash

git clone http://git.code.sf.net/p/libsharp/code libsharp-code
cd libsharp-code
autoconf && ./configure --enable-pic --disable-openmp && make
cd ..
git clone https://github.com/mselig/libsharp-wrapper libsharp-wrapper
cd libsharp-wrapper
python setup.py build_ext install
cd ..
rm -r libsharp-code
rm -r libsharp-wrapper

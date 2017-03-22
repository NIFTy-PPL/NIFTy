#!/bin/bash

cd test/data
for filename in *.py; do
  ./$filename
done
cd ../..

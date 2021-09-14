#!/usr/bin/env bash

# Strictly disallow uninitialized Variables
set -u
# Exit if a single command breaks and its failure is not handled accordingly
set -e

# Save current path to get back later
previous_pwd="$(pwd)"
cd "$(git rev-parse --show-toplevel)"

sphinx-apidoc -e -o doc/source/mod jifty1
sphinx-build -b html doc/source/ doc/build/

# Restore previous path
cd "${previous_pwd}"

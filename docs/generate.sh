rm -rf docs/build docs/source/mod
better-apidoc -l -e -d 2 -t docs/generation-templates -o docs/source/mod nifty4
sphinx-build -b html docs/source/ docs/build/

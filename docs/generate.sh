rm -rf docs/build docs/source/mod
sphinx-apidoc -l -e -d 2 -o docs/source/mod nifty4
#python docs/better_apidoc.py -l -e -d 3 -t docs/generation-templates -o docs/source/mod nifty4
sphinx-build -b html docs/source/ docs/build/

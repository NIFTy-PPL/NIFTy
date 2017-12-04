ln -s nifty nifty2go
rm -rf docs/build docs/source/mod
sphinx-apidoc -l -e -d 3 -o docs/source/mod nifty2go
sphinx-build -b html docs/source/ docs/build/
rm nifty2go

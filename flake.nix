{
  description = "Numerical Information Field Theory";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        myPyPkgs = pkgs.python3Packages;

        version = "8.5.7";

        req.minimal = with myPyPkgs; [ numpy scipy ducc0 ];
        req.dev = with myPyPkgs; [ pytest pytest-cov matplotlib ];
        req.mpi = [ myPyPkgs.mpi4py ];
        req.jax = with myPyPkgs; [ jax jaxlib ];
        req.rest = with myPyPkgs; [ astropy h5py ];
        req.docs = with myPyPkgs; [
          sphinx
          pkgs.jupyter # python3Packages.jupyter is broken, see https://github.com/NixOS/nixpkgs/issues/299385
          jupytext
          pydata-sphinx-theme
          myst-parser
        ];
        allreqs = pkgs.lib.attrValues req;

        nifty = myPyPkgs.buildPythonPackage {
          pname = "nifty8";
          inherit version;
          src = ./.;
          pyproject = true;
          build-system = with pkgs.python3.pkgs; [ setuptools ];
          dependencies = req.minimal ++ req.mpi;
          checkInputs = with myPyPkgs; [ pytestCheckHook pytest-xdist ]
            ++ allreqs;
          disabledTestPaths = [ "test/test_re" ];
          postCheck = ''
            ${
              pkgs.lib.getExe' pkgs.mpi "mpirun"
            } -n 2 --bind-to none python3 -m pytest test/test_mpi
          '';
          pythonImportsCheck = [ "nifty8" ];
        };

        nifty-docs = pkgs.stdenv.mkDerivation {
          name = "nifty-docs";
          inherit version;
          src = ./.;
          buildInputs = allreqs ++ [ nifty ];
          buildPhase = "sh docs/generate.sh";
          installPhase = ''
            mkdir $out
            mv docs/build/* $out
          '';
        };

      in {
        # Standard nifty package
        packages.default = nifty;

        # Build nifty docs (`nix build .#docs`)
        packages."docs" = nifty-docs;

        # Development shell (`nix develop .`) including python-lsp-server for development
        devShells.default = pkgs.mkShell {
          buildInputs = allreqs ++ (with myPyPkgs; [
            pip
            venvShellHook
            python-lsp-server
            python-lsp-ruff
          ]) ++ (with pkgs; [ ruff ruff-lsp ]);
          venvDir = ".nix-nifty-venv";

          shellHook = ''
            export PIP_PREFIX=$(pwd)/_build/pip_packages
            export PYTHONPATH="$PIP_PREFIX/${myPyPkgs.python.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            unset SOURCE_DATE_EPOCH
          '';
        };
      });
}

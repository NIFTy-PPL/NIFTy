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
        pkgs-cuda = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        myPyPkgs = pkgs.python3Packages;
        myPyPkgs-cuda = pkgs-cuda.python3Packages;

        version = "9.0.0";

        req.minimal = with myPyPkgs; [ numpy scipy ducc0 ];
        req.dev = with myPyPkgs; [ pytest pytest-cov pytest-xdist matplotlib ];
        req.mpi = [ myPyPkgs.mpi4py pkgs.openmpi pkgs.openssh ];
        req.jax = with myPyPkgs; [ jax jaxlib ];
        req.rest = with myPyPkgs; [ astropy h5py ];
        req.docs = with myPyPkgs; [
          sphinx
          pkgs.jupyter # python3Packages.jupyter is broken, see https://github.com/NixOS/nixpkgs/issues/299385
          jupytext
          pydata-sphinx-theme
          sphinxcontrib-bibtex
          myst-parser
        ];

        req-cuda.minimal = with myPyPkgs-cuda; [ numpy scipy ducc0 cupy pkgs-cuda.cudaPackages.cudatoolkit ];
        req-cuda.dev = with myPyPkgs-cuda; [ pytest pytest-cov pytest-xdist matplotlib ];
        req-cuda.mpi = [ myPyPkgs-cuda.mpi4py pkgs.openmpi pkgs.openssh ];
        req-cuda.jax = with myPyPkgs-cuda; [ jax jaxlib ];
        req-cuda.rest = with myPyPkgs-cuda; [ astropy h5py ];
        req-cuda.docs = with myPyPkgs-cuda; [
          sphinx
          pkgs-cuda.jupyter # python3Packages.jupyter is broken, see https://github.com/NixOS/nixpkgs/issues/299385
          jupytext
          pydata-sphinx-theme
          sphinxcontrib-bibtex
          myst-parser
        ];

        allreqs = pkgs.lib.attrValues req;
        allreqs-cuda = pkgs.lib.attrValues req-cuda;

        nifty = myPyPkgs.buildPythonPackage {
          pname = "nifty";
          inherit version;
          src = ./.;
          pyproject = true;
          build-system = with pkgs.python3.pkgs; [ setuptools ];
          dependencies = req.minimal ++ req.mpi;
          checkInputs = with myPyPkgs; [ pytestCheckHook pytest-xdist ]
            ++ allreqs;
          postCheck = ''
            ${
              pkgs.lib.getExe' pkgs.mpi "mpirun"
            } -n 2 --bind-to none python3 -m pytest test/test_cl/test_mpi
          '';
          pythonImportsCheck = [ "nifty" ];
        };
        nifty-cuda = myPyPkgs-cuda.buildPythonPackage {
          pname = "nifty";
          inherit version;
          src = ./.;

          pyproject = true;
          build-system = with pkgs-cuda.python3.pkgs; [ setuptools ];
          dependencies = req-cuda.minimal ++ req-cuda.mpi;
          checkInputs = with myPyPkgs-cuda; [ pytestCheckHook pytest-xdist ]
            ++ allreqs;
          postCheck = ''
            ${
              pkgs-cuda.lib.getExe' pkgs-cuda.mpi "mpirun"
            } -n 2 --bind-to none python3 -m pytest test/test_mpi
          '';
          pythonImportsCheck = [ "nifty" ];
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
        packages."cuda" = nifty-cuda;

        # Build nifty docs (`nix build .#docs`)
        packages."docs" = nifty-docs;

        # Run `nix develop .` to enter the development shell including, e.g.,
        # python-lsp-server. Then compile nifty with, e.g., `pip3 install .`
        devShells.default = pkgs.mkShell {
          buildInputs = allreqs ++ (with myPyPkgs; [
            pip
            venvShellHook
            python-lsp-server
            python-lsp-ruff
            pkgs.ruff
          ]);
          venvDir = ".nix-nifty-venv";

          shellHook = ''
            export PIP_PREFIX=$(pwd)/_build/pip_packages
            export PYTHONPATH="$PIP_PREFIX/${myPyPkgs.python.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            unset SOURCE_DATE_EPOCH
          '';
        };

        devShells."cuda" = pkgs-cuda.mkShell {
          buildInputs = allreqs-cuda ++ (with myPyPkgs-cuda; [
            pip
            venvShellHook
            python-lsp-server
            python-lsp-ruff
            pkgs-cuda.ruff
          ]);
          venvDir = ".nix-nifty-cuda-venv";

          CUDA_HOME = "/usr/local/cuda";
          LD_LIBRARY_PATH =
            "${pkgs-cuda.cudaPackages.cudatoolkit.lib}/lib:/usr/local/cuda/lib64";

          shellHook = ''
            export PIP_PREFIX=$(pwd)/_build/pip_packages
            export PYTHONPATH="$PIP_PREFIX/${myPyPkgs-cuda.python.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            unset SOURCE_DATE_EPOCH
          '';
        };

      });
}

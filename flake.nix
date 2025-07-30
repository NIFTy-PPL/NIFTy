{
  description = "Numerical Information Field Theory";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgsBase = import nixpkgs { inherit system; };
        pkgsCuda = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        version = "9.1.0";

        getDeps = pkgs: cudaSupport:
          let
            pyPkgs = pkgs.python3Packages;
            main = with pyPkgs;
              [ numpy scipy ducc0 ] ++ (if cudaSupport then [
                cupy
                pkgs.cudaPackages.cudatoolkit
              ] else
                [ ]) ++ [ pyPkgs.mpi4py pkgs.openmpi pkgs.openssh ];
          in {
            pyPkgs = pyPkgs;
            main = main;
            test = main ++ (with pyPkgs; [
              pytestCheckHook
              pytest
              pytest-cov
              pytest-xdist
              matplotlib
              h5py
            ]);
            docs = [
              pkgs.jupyter # python3Packages.jupyter is broken, see https://github.com/NixOS/nixpkgs/issues/299385
              pyPkgs.jupytext
              pyPkgs.pydata-sphinx-theme
              pyPkgs.sphinx
              pyPkgs.sphinxcontrib-bibtex
              pyPkgs.myst-parser
              pyPkgs.jax
              pyPkgs.matplotlib
            ];
          };

        mkNifty = { cudaSupport ? false, doCheck ? true }:
          let
            pkgs = if cudaSupport then pkgsCuda else pkgsBase;
            deps = getDeps pkgs cudaSupport;
          in pkgs.lib.makeOverridable (args:
            deps.pyPkgs.buildPythonPackage rec {
              pname = "nifty";
              inherit version;
              src = ./.;
              pyproject = true;
              build-system = with pkgs.python3.pkgs; [ setuptools ];
              dependencies = deps.main;
              doCheck = args.doCheck or true;
              checkInputs = if doCheck then deps.test else [ ];
              pytestFlagsArray = [ "--ignore=test/test_re" ];
              postCheck = pkgs.lib.optionalString doCheck ''
                ${
                  pkgs.lib.getExe' pkgs.mpi "mpirun"
                } -n 2 --bind-to none python3 -m pytest test/test_cl/test_mpi
              '';
              pythonImportsCheck = [ "nifty" ];
            }) { # TODO: Is this really the way to go in nix? (see also below)
              cudaSupport = cudaSupport;
              doCheck = doCheck;
            };

        mkNiftyDocs = { cudaSupport ? false }:
          let
            pkgs = if cudaSupport then pkgsCuda else pkgsBase;
            deps = getDeps pkgs cudaSupport;
          in pkgs.lib.makeOverridable (args:
            pkgs.stdenv.mkDerivation {
              name = "nifty-docs";
              inherit version;
              src = ./.;
              buildInputs = deps.docs
                ++ (if cudaSupport then [ nifty-cuda ] else [ nifty ]);
              buildPhase = "sh docs/generate.sh";
              installPhase = ''
                mkdir $out
                mv docs/build/* $out
              '';
            }) { cudaSupport = cudaSupport; };

        mkDevShell = { cudaSupport ? false }:
          let
            pkgs = if cudaSupport then pkgsCuda else pkgsBase;
            deps = getDeps pkgs cudaSupport;
            cudaEnv = if cudaSupport then ''
              export CUDA_HOME=${toString pkgs.cudaPackages.cudatoolkit}
              export LD_LIBRARY_PATH=${
                toString pkgs.cudaPackages.cudatoolkit.lib
              }/lib:$LD_LIBRARY_PATH
            '' else
              "";
            #   CUDA_HOME = "/usr/local/cuda";
            #   LD_LIBRARY_PATH =
            #     "${pkgs-cuda.cudaPackages.cudatoolkit.lib}/lib:/usr/local/cuda/lib64";
          in pkgs.mkShell {
            buildInputs = deps.main ++ deps.test ++ (with deps.pyPkgs; [
              pip
              venvShellHook
              python-lsp-server
              python-lsp-ruff
            ]) ++ [ pkgs.ruff ];
            venvDir = ".nix-nifty-venv";
            shellHook = ''
              export PIP_PREFIX=$(pwd)/_build/pip_packages
              export PYTHONPATH="$PIP_PREFIX/${deps.pyPkgs.python.sitePackages}:$PYTHONPATH"
              export PATH="$PIP_PREFIX/bin:$PATH"
              unset SOURCE_DATE_EPOCH
              ${cudaEnv}
            '';
          };

        nifty = mkNifty { cudaSupport = false; };
        nifty-cuda = mkNifty { cudaSupport = true; };
      in {
        # Standard nifty package
        packages.default = nifty;
        packages."cuda" = nifty-cuda;

        # Build nifty docs (`nix build .#docs`)
        packages."docs" = mkNiftyDocs { cudaSupport = false; };

        # Run `nix develop .` to enter the development shell including, e.g.,
        # python-lsp-server. Then compile nifty with, e.g., `pip3 install .`
        devShells.default = mkDevShell { cudaSupport = false; };
        devShells.cuda = mkDevShell { cudaSupport = true; };

      });
}

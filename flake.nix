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

        version = "8.5.2";

        req.minimal = with myPyPkgs; [ numpy scipy ducc0 ];
        req.dev = with myPyPkgs; [ pytest pytest-cov matplotlib ];
        # req.mpi = [ myPyPkgs.mpi4py pkgs.openmpi pkgs.openssh ];
        req.jax = with myPyPkgs; [ jax jaxlib ];
        req.rest = with myPyPkgs; [ astropy h5py ];
        req.docs = with myPyPkgs; [
          sphinx
          pkgs.jupyter  # python3Packages.jupyter is broken, see https://github.com/NixOS/nixpkgs/issues/299385
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
          dependencies = req.minimal;

          # TODO Add MPI tests
          checkInputs =  [ myPyPkgs.pytestCheckHook ] ++ allreqs;
          disabledTestPaths = [ "test/test_mpi" ];
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
          nativeBuildInputs =
            (with myPyPkgs; [ python-lsp-server python-lsp-ruff ])
            ++ (with pkgs; [ ruff ruff-lsp ]) ++ allreqs;
        };

        # Shell in which nifty is installed (`nix develop .#nifty-installed`),
        # e.g., for building the docs (`sh docs/generate.sh`) or running demos
        devShells."nifty-installed" = pkgs.mkShell {
          nativeBuildInputs = allreqs;
          packages = [ nifty ];
        };

      });
}

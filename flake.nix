{
  description = "Numerical Information Field Theory";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        myPyPkgs = pkgs.python3Packages;

        req.minimal = with myPyPkgs; [ numpy scipy ducc0 ];
        req.dev = with myPyPkgs; [ pytest pytest-cov matplotlib ];
        req.mpi = [ myPyPkgs.mpi4py pkgs.openmpi pkgs.openssh ];
        req.jax = with myPyPkgs; [ jax jaxlib ];
        req.rest = with myPyPkgs; [ astropy ];

        req.docs = with myPyPkgs; [ sphinx jupyter jupytext ];
        # TODO add pydata-sphinx-theme
      in {
        packages.default = myPyPkgs.buildPythonPackage {
          pname = "nifty8";
          version = "8.0";  # TODO Set this automatically
          src = ./.;
          nativeBuildInputs = req.minimal;
          checkInputs =  [ myPyPkgs.pytestCheckHook ] ++ req.dev ++ req.mpi;
          pytestFlagsArray = [ "test" ];
          pythonImportsCheck = [ "nifty8" ];
        };

        # TODO Add version with MPI, jax and both

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with myPyPkgs; [ pip venvShellHook ] ++ req.minimal ++ req.dev;
          # ( pkgs.lib.attrValues req ) ;
          venvDir = "./.nix-nifty-venv";
        };
      });
}

FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip python3-venv \
    # Documentation build dependencies
    dvipng texlive-latex-base texlive-latex-extra pandoc \
    # Dependency of mpi4py
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash runner
USER runner
WORKDIR /home/runner

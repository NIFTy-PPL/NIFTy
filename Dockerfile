FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Documentation build dependencies
    dvipng texlive-latex-base texlive-latex-extra pandoc \
    # Dependency of mpi4py
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages pip-tools \
    # Source all dependencies from the pyprojects.toml
    && pip-compile --all-extras --output-file=requirements.txt pyproject.toml \
    && pip install --break-system-packages -r requirements.txt

# Set matplotlib backend
ENV MPLBACKEND agg

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser

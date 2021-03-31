FROM debian:testing-slim
RUN apt-get update && apt-get install -qq \
      # General environment
      git python3-pip python3-pytest \
      && apt-get update \
      && apt-get install -qq \
      # jifty dependencies
      python3-scipy \
      # Documentation build dependencies
      dvipng texlive-latex-base texlive-latex-extra \
      && rm -rf /var/lib/apt/lists/*
RUN pip install sphinx pydata-sphinx-theme jax jaxlib

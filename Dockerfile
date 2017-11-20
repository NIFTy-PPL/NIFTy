#FROM ubuntu:artful
FROM debian:testing-slim

# dependencies via apt
RUN apt-get update
ADD ci/install_basics.sh /tmp/install_basics.sh
RUN sh /tmp/install_basics.sh


# python dependencies
ADD ci/requirements.txt /tmp/requirements.txt
RUN pip install --process-dependency-links -r /tmp/requirements.txt


# copy sources and install nifty
COPY . /tmp/NIFTy
RUN pip install /tmp/NIFTy


# Cleanup
RUN rm -r /tmp/*

# 2018 Spring astro hack
A repository for the 2018 Spring semester Society of Physics Students (SPS)
and Rutgers Astronomical Society (RAS) astro hack sessions.

## Getting started
Please download or clone the repository using:

    git clone https://github.com/rutgers-physics-ml/astro-hack-2018-spring


You will need Anaconda Python 3.6 or later, and we will supply all
necessary packages. Note that you will need an internet connection.
First add the `conda-forge` channel for access to some necessary
Python packages:

    conda config --add channels conda-forge

Install the rest of the packages by running:

    conda env create -f environment.yml

These packages have now been installed on your system, to a virtual
environment in Anaconda Python named `astrohack`.

## Python virtual environments
You will need to *activate* the `astrohack` environment before executing any
code or serving any Jupyter notebooks. This can be accomplished by running:

    source activate astrohack

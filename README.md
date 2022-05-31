 [![pipeline status](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/pipeline.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![coverage report](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/coverage.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![Latest Release](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/badges/release.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/releases)

# Learning in Sequential Bayesian Games

This project implements sequential auctions as a Mulit-Agent Reinforcement Learning environment.
It uses the implementation of Alexander Neitz's SimpleSoccer environment as basis.

## Setup

Note: These setup instructions assume a linux-based OS and uses python 3.8.10 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)

`sudo apt-get install virtualenv`

Create a virtual environment with virtual env (you can also choose your own name)

`virtualenv sequential-auctions-on-gpu`

You can specify the python version for the virtual environment via the -p flag. Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3 sequential-auctions-on-gpu` uses the standard python3 version from the system).

activate the environment with

`source ./sequential-auctions-on-gpu/bin/activate`

Install all requirements

`pip install -r requirements.txt`

## Install pre-commit hooks (for development)
Install pre-commit hooks for your project

`pre-commit install`

Verify by running on all files:

`pre-commit run --all-files`

For more information see https://pre-commit.com/.
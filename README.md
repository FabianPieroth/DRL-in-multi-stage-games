 [![pipeline status](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/pipeline.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![coverage report](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/coverage.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![Latest Release](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/badges/release.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/releases)

# Learning in Continuous Sequential Bayesian Games

This project implements a set of sequential games with continuous state and action spaces and makes use of multi-agent reinforcement learning to compute optimal strategies. In particular we look at *sequential sales* [(Krishna 2003, Chapter 15)](https://www.sciencedirect.com/book/9780124262973/auction-theory) and a *signaling contest* [(Zhang, 2008)](https://ideas.repec.org/p/qed/wpaper/1184.html) market. Some smaller toy examples, such as a soccer simulation (credit to Alexander Neitz) and rock-paper-scissors are also implemented. The base algorithms are vendored from [StableBaselines3](https://github.com/DLR-RM/stable-baselines3).


## Features
* Multi-agent learning
* Support for continuos (and discrete) state and action spaces
* High degree of parallelization
* Brute force verifier that checks for optimality

Implemented are the following algorithms: REINFORCE, PPO, Deep Q-learning.

## Limitations
* Only supports simultaneous move games
* Only supports fixed length games


# Installation

## Setup

Note: These setup instructions assume a linux-based OS and uses python 3.8.10 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)
`sudo apt-get install virtualenv`

Create a virtual environment with virtual env (you can also choose your own name)
`virtualenv sequential-auctions-on-gpu`

You can specify the python version for the virtual environment via the -p flag. Note that this version already needs to be installed on the system (e.g., `virtualenv - p python3 sequential-auctions-on-gpu` uses the standard python3 version from the system).

Activate the environment with
`source ./sequential-auctions-on-gpu/bin/activate`

Install all requirements
`pip install -r requirements.txt`


## Install pre-commit hooks (for development)

Install pre-commit hooks for your project
`pre-commit install`

Verify by running on all files:
`pre-commit run --all-files`

For more information see https://pre-commit.com/.


# Maintainers and suggested citation

This project is maintained by Fabian Pieroth [(@FabianPieroth)](https://github.com/FabianPieroth) and Nils Kohring [(@kohring)](https://github.com/kohring).

If you find this repository helpful and use it in your work, please consider using the following citation:

```
@misc{dss2023,
  author = {Bichler, Martin and Kohring, Nils and Pieroth, Fabian},
  title = {Learning in Continuous Sequential Bayesian Games},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/heidekrueger/bnelearn}}
}
````

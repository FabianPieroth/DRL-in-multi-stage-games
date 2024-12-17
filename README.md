 [![pipeline status](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/pipeline.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![coverage report](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/badges/main/coverage.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/commits/main) [![Latest Release](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/badges/release.svg)](https://gitlab.lrz.de/ge69hoj/sequential-auction-on-gpu/-/releases)


---
# Learning in Continuous Sequential Bayesian Games

This project implements a set of sequential games with continuous state and action spaces and makes use of multi-agent reinforcement learning to compute optimal strategies. In particular we look at sequential sales [(Krishna 2003, Chapter 15)](https://www.sciencedirect.com/book/9780124262973/auction-theory) and a signaling contest [(Zhang, 2008)](https://ideas.repec.org/p/qed/wpaper/1184.html) market. Some smaller toy examples, such as a soccer simulation (credit to Alexander Neitz) and rock-paper-scissors are also implemented.

The base algorithms are vendored from [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) and the environment interface is a multiagent extension of [OpenAI's gym](https://github.com/openai/gym) framework.


## Features
* Multi-agent learning
* Support for continuos (and discrete) state and action spaces
* High degree of parallelization
* Brute force verifier that checks for optimality

Implemented are the following algorithms: 

| Algorithm        | Continuous | Discrete |
|------------------|------------|----------|
| REINFORCE        |          ☑|        ☑ |
| PPO              |          ☑|        ☑ |
| Deep Q-learning  |          ☐|        ☑ |
| TD3              |          ☑|        ☐ |

## Limitations
* Only supports simultaneous move games
* Only supports fixed length games


---
# Installation

## Setup

Note: These setup instructions assume a linux-based OS and uses python 3.9.21.

If necessary, install `virtualenv` (or whatever you prefer for virtual envs)
```bash
sudo apt-get install virtualenv
```

Create a virtual environment with virtual env (you can also choose your own name)
```bash
virtualenv sequential-auctions-on-gpu
```

You can specify the python version for the virtual environment via the -p flag. Note that this version already needs to be installed on the system (e.g., "virtualenv sequential-auctions-on-gpu -p `which python3.XX`" uses the standard python3 version from the system).

Activate the environment with
```bash
source ./sequential-auctions-on-gpu/bin/activate
```

Due to older dependencies, setuptools and pip need to have a specific version as well. (See https://stackoverflow.com/questions/77124879/pip-extras-require-must-be-a-dictionary-whose-values-are-strings-or-lists-of for more information.) Therefore, check or run:
```bash
pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
```

Install all requirements
```bash
pip install -r requirements.txt
```


## Install pre-commit hooks (for development)

Install pre-commit hooks for your project via
```bash
pre-commit install
```

and verify by running on all files
```bash
pre-commit run --all-files
```

For more information see [https://pre-commit.com/](https://pre-commit.com/).


---
# Getting started

The main experiments are located and can be run from `scripts/run_experiments.py`. The workflow is as follows:
1. From the `utils` sub-package, import `io_ut` for handling the configuration of experiments (which game, algorithms, hyperparameters, etc.), and `coord_ut` which handles and runs the learning.
2. Overwrite any of the default configuration settings via `io_ut.get_config()`. Check out the `configs/` folder for the options available and note that hierarchical parameters can be accessed via dots, e.g. `"rl_envs.num_agents=2"`.
3. Start learning via calling `coord_ut.start_ma_learning(config)`.

A minimal working example defaults to learning in sequential sales where all agents learn via PPO (see `configs/config.yaml`):
```python
import src.utils.io_utils as io_ut
import src.utils.coordinator_utils as coord_ut

config = io_ut.get_config(overrides=["rl_envs.num_agents=2"])
coord_ut.start_ma_learning(config)
```

## Adding a new environment (game)
Any new environment must inherit from the `BaseEnvForVec` class which supplies a standardized interface to other parts of this framework via its abstract methods.

### 1. New environment class

The high-level work flow is as follows:
1. The number of agents, the observation spaces for the agents, and their action spaces must be defined via `_get_num_agents`, `_init_observation_spaces`, and `_init_action_spaces`, respectively.
2. In `sample_new_states`, it is defined how the game states are initialized. (The function takes a batch size as argument for the number of games that are played in parallel.)
3. The agents/learners will access their current observations via `get_observations`.
4. They publish their actions via `compute_step` which also takes the current states as argument. It returns the new `observations`, `rewards`, `dones` (an indicator of which games have reached a terminal state), and the new states `new_states`.

```python
from typing import Any, Dict
from gym.spaces import Space
from src.envs.torch_vec_env import BaseEnvForVec


class YourEnvironment(BaseEnvForVec):

    def __init__(self, config: Dict, device):
        super().__init__(config, device)

    def _get_num_agents(self) -> int:
        pass

    def _init_observation_spaces(self) -> Dict[int, Space]:
        pass

    def _init_action_spaces(self) -> Dict[int, Space]:
        pass

    def to(self, device) -> Any:
        pass

    def sample_new_states(self, n: int) -> torch.Tensor:
        pass

    def compute_step(self, cur_states, actions: torch.Tensor):
        pass

    def get_observations(self, states) -> torch.Tensor:
        pass

    def render(self, state):
        pass
```

For more details, check out the doc-strings of the abstract methods or for an example one of the implemented games such as `RockPaperScissors`.

__Logging:__ During learning, one may want to log and/or evaluate the learning progress. That's what the function `custom_evaluation` is for, which is called regularly during the MARL learning procedure.

__Type of observation/action spaces:__ This framework is primarily focused on games with continuous observation/state and action spaces. However, due to many algorithms being limited to finite spaces, we have implemented a so-called `SpaceTranslator` that can discretize continuous games. See `src/envs/space_translators.py`.

### 2. Register the new environment

__Add environment to configurations:__ Create a new configuration file with your environment's name `<env-name>` in the directory `configs/rl_envs/`. This should include any changeable parameters of the game, such as the number of agents or the payment rule for auctions.

__Add environment to environment selection:__ Import your environment class in `src.utils.coordinator_utils.py`. Add another `elif`-case that initializes your new learner in the function `get_env()`.

## Adding a new learner

### 1. New learner class

There are in general two ways to create a new learner.

__Custom learner:__ Inherit from `MABaseAlgorithm` in `src/learners/base_learner.py`. Then overwrite the specified methods accordingly. See `src/learners/random_learner.py` as example.

__Adapt StableBaselines3 algorithm:__ Inherit from `BaseAlgorithm` from `stable_baselines3.common.base_class` or one of its super classes. Furthermore, add the methods that are added in `MABaseAlgorithm`. Additionally, one needs to rewrite internal logic so that the data received by `ingest_data_into_learner` is sufficient for training. See `VecPPO` or `GPUDQN` as example.

### 2. Register the new learner

There are several steps needed to register a new learner/algorithm to the framework.

__Add learner to configurations:__ Create a new folder with your algorithm's name `<algo-name>` in `configs/algorithm_configs`. Add a file named `<algo-name>_default.yaml` that includes configurations to your algorithm. This file will be passed into the learner during init. Add the line `- <algo-name>: <algo-name>_default` to `configs/algorithm_configs/all_algos.yaml`. See the `RandomLearner` as minimal example.

__Add learner to algorithm selection:__ Import your learner class in `src.utils.policy_utils.py`. Add another `elif`-case that initializes your new learner. See `RandomLearner` as minimal example.

---
# Maintainers and suggested citation

This project is maintained by Fabian Pieroth ([@FabianPieroth](https://github.com/FabianPieroth)) and Nils Kohring ([@kohring](https://github.com/kohring)).

If you find this repository helpful and use it in your work, please consider using the following citation:

```bibtex
@misc{dss2023,
  author = {Bichler, Martin and Kohring, Nils and Pieroth, Fabian},
  title = {Learning in Continuous Sequential Bayesian Games},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/404}}
}
```

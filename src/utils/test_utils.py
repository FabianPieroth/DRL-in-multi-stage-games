import copy

import hydra

import src.utils.coordinator_utils as coord_ut


def run_limited_learning(config):
    """Runs multi agent learning for `config`."""
    config = copy.deepcopy(config)

    ma_learner = coord_ut.start_ma_learning(config)
    hydra.core.global_hydra.GlobalHydra().clear()
    return ma_learner

"""This module tests the individual space translators."""
import hydra
import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete, MultiDiscrete

import src.utils_folder.io_utils as io_ut
import src.utils_folder.test_utils as tst_ut
from src.envs.space_translators import (
    BoxToDiscreteSpaceTranslator,
    IdentitySpaceTranslator,
    MultiDiscreteToDiscreteSpaceTranslator,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "CPU"

ids_id, testdata_id = zip(
    *[
        [
            "discrete_identity",
            (Discrete(32), torch.randint(low=0, high=32, size=(11,))),
        ],
        [
            "multidiscrete_identity",
            (
                MultiDiscrete(tuple([3 for _ in range(13)])),
                torch.randint(low=0, high=3, size=(5, 13)),
            ),
        ],
        [
            "single_random_identity",
            (Box(low=-np.inf, high=np.inf, shape=(1,)), torch.normal(0, 1, size=(41,))),
        ],
        [
            "multi_random_identity",
            (
                Box(low=-np.inf, high=np.inf, shape=(17,)),
                torch.normal(0, 1, size=(41, 17)),
            ),
        ],
    ]
)


@pytest.mark.parametrize("domain_space, in_tensor", testdata_id, ids=ids_id)
def test_identity_space_translator(domain_space, in_tensor):
    in_tensor.to(DEVICE)
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)
    config = {}
    translator = IdentitySpaceTranslator(domain_space=domain_space, config=config)
    assert torch.all(
        translator.translate(in_tensor) == in_tensor
    ).item(), "The identity translation changes the input tensor!"
    assert torch.all(
        translator.inv_translate(in_tensor) == in_tensor
    ).item(), "The inverse identity translation changes the input tensor!"
    assert torch.all(
        translator.inv_translate(translator.translate(in_tensor)) == in_tensor
    ).item(), "The translation is not invertible!"


ids_md_to_d, testdata_md_to_d = zip(
    *[
        [
            "multidiscrete_uniform",
            (
                MultiDiscrete(tuple([3 for _ in range(4)])),
                torch.tensor([[0, 1, 2, 0], [2, 1, 0, 0], [0, 0, 0, 2]]),
                torch.tensor([15, 63, 2]),
            ),
        ],
        [
            "multidiscrete_asymmetric",
            (
                MultiDiscrete((3, 4, 5)),
                torch.tensor([[1, 3, 4], [0, 2, 1], [2, 1, 0], [0, 0, 4]]),
                torch.tensor([39, 11, 45, 4]),
            ),
        ],
    ]
)


@pytest.mark.parametrize(
    "domain_space, in_tensor, out_tensor", testdata_md_to_d, ids=ids_md_to_d
)
def test_md_to_d_space_translator_invertibility_md_d(
    domain_space, in_tensor, out_tensor
):
    in_tensor.to(DEVICE)
    out_tensor.to(DEVICE)
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)
    config = {"multi_space_shape": tuple(domain_space.nvec)}
    translator = MultiDiscreteToDiscreteSpaceTranslator(
        domain_space=domain_space, config=config
    )
    assert torch.all(
        translator.translate(in_tensor) == out_tensor
    ).item(), "The translation does not produce the expected output!"
    assert torch.all(
        translator.inv_translate(out_tensor) == in_tensor
    ).item(), "The inverse translation does not produce the expected result!"
    assert torch.all(
        translator.inv_translate(translator.translate(in_tensor)) == in_tensor
    ).item(), "The translation is not invertible!"


ids_box_to_d, testdata_box_to_d = zip(
    *[
        [
            "box_to_d_bounded",
            (
                Box(low=0.0, high=1.0, shape=(1,)),
                11,
                2.0,
                torch.tensor([-1.0, 0.0, 0.02, 0.08, 0.81, 0.91, 0.99, 1.2]),
                torch.tensor([0, 0, 0, 1, 8, 9, 10, 10]),
            ),
        ],
        [
            "box_to_d_lower_bounded",
            (
                Box(low=0.0, high=np.inf, shape=(1,)),
                11,
                1.0,
                torch.tensor([-1.0, 0.0, 0.02, 0.08, 0.81, 0.91, 0.99, 1.2]),
                torch.tensor([0, 0, 0, 1, 8, 9, 10, 10]),
            ),
        ],
        [
            "box_to_d_unbounded",
            (
                Box(low=-np.inf, high=np.inf, shape=(1,)),
                21,
                2.0,
                torch.tensor([-1.0, 0.0, 0.02, 0.08, 0.81, 0.91, 0.99, 1.2]),
                torch.tensor([0, 10, 10, 11, 18, 19, 20, 20]),
            ),
        ],
    ]
)


@pytest.mark.parametrize(
    "domain_space, granularity, maximum_width, in_tensor, out_tensor",
    testdata_box_to_d,
    ids=ids_box_to_d,
)
def test_box_to_d_space_translator(
    domain_space, granularity, maximum_width, in_tensor, out_tensor
):
    in_tensor.to(DEVICE)
    out_tensor.to(DEVICE)
    hydra.core.global_hydra.GlobalHydra().clear()
    io_ut.set_global_seed(0)
    config = {"granularity": granularity, "maximum_width": maximum_width}
    translator = BoxToDiscreteSpaceTranslator(domain_space=domain_space, config=config)
    assert torch.all(
        translator.translate(in_tensor) == out_tensor
    ).item(), "The translation does not produce the expected output!"

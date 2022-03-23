"""Utilities for our custom learners"""
import os

import torch as th


def explained_variance(y_pred: th.Tensor, y_true: th.Tensor) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = th.var(y_true)
    return (
        float("nan")
        if var_y == 0
        else (1 - th.var(y_true - y_pred) / var_y).cpu().item()
    )


def new_log_path(path: str):
    """Increment path counter"""
    i = 1
    while os.path.isdir(f"{path}_{i}"):
        i += 1
    new_path = f"{path}_{i}"
    os.makedirs(new_path)
    return new_path


def tensor_norm(b1: th.Tensor, b2: th.Tensor, p: float = 2) -> float:
    r"""Calculates the approximate "mean" Lp-norm between two action vectors.
    Source: https://gitlab.lrz.de/heidekrueger/bnelearn/-/blob/master/bnelearn/util/metrics.py

    .. math::
        \sum_i=1^n(1/n * |b1 - b2|^p)^{1/p}

    If p = Infty, this evaluates to the supremum.
    """
    assert b1.shape == b2.shape

    if p == float("Inf"):
        return (b1 - b2).abs().max()

    # finite p
    n = float(b1.shape[0])

    # calc. norm & detach for disregarding any gradient info
    return (th.dist(b1, b2, p=p) * (1.0 / n) ** (1 / p)).detach().item()


def batched_index_select(input: th.Tensor, dim: int, index: th.Tensor) -> th.Tensor:
    """
    Extends the torch `index_select` function to be used for multiple batches
    at once.

    This code is borrowed from https://discuss.pytorch.org/t/batched-index-select/9115/11.

    author:
        dashesy

    args:
        input: Tensor which is to be indexed
        dim: Dimension
        index: Index tensor which proviedes the seleting and ordering.

    returns:
        Indexed tensor
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)

    return th.gather(input, dim, index)

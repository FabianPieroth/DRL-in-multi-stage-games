"""Utilities for our custom learners"""
import torch as th


def explained_variance(y_pred: th.Tensor, y_true: th.Tensor) -> th.Tensor:
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
    return th.nan if var_y == 0 else 1 - th.var(y_true - y_pred) / var_y

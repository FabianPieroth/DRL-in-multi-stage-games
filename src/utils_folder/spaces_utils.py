from typing import List, Tuple, Union

import torch

Tensor = torch.Tensor
Shape = Union[List[int], Tuple[int, ...], torch.Size]


def ravel_multi_index(coords: Tensor, shape: Shape) -> Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.
    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def unravel_index(indices: Tensor, shape: Shape) -> Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.
    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]


def discrete_to_multidiscrete(int_actions: Tensor, array_shape: Tuple[int]) -> Tensor:
    """
    Args:
        nodes (Tensor): shape=(x, )
        array_shape (Tuple[int]): the multi-index to ravel into

    Returns:
        Tensor: shape(x, len(array_shape))
    """
    return unravel_index(int_actions, array_shape)


def multidiscrete_to_discrete(multi_actions: Tensor, array_shape: Tuple[int]) -> Tensor:
    """Tensor of shape (batch_size, num_multi_discrete_actions) is turned into a raveled multi-index by array_shape shape (y,)
    Args:
        actions (Tensor): shape (batch_size, len(array_shape))
        array_shape (Tuple[int]): shape(batch_size, )

    Returns:
        Tensor: shape(batch_size, )
    """
    return ravel_multi_index(multi_actions, array_shape)

from typing import List

import torch


def repeat_tensor_along_new_axis(
    data: torch.Tensor, pos: List[int], repeats: List[int]
) -> torch.Tensor:
    """Add additional dimensions as pos and repeat it for repeats along these dimensions.

    Args:
        data (torch.Tensor): tensor to be repeated
        pos (List[int]): strictly increasing order of positions where repeats should be in out-tensor
        repeats (List[int]): number of repeats of dimensions

    Returns:
        torch.Tensor: repeated tensor
    Example:
    data.shape = (2, 3)
    pos = [1, 2]
    repeats = [11, 7]
    out.shape = (2, 11, 7, 3)
    """
    # TODO: Check if torch.expand() instead of torch.repeat() is feasible!
    assert len(pos) == len(repeats), "Each pos needs a specified repeat!"
    for single_pos in pos:
        data = data.unsqueeze(single_pos)
    dims_to_be_repeated = [1 for i in range(len(data.shape))]
    for k, repeat in enumerate(repeats):
        dims_to_be_repeated[pos[k]] = repeat
    return data.repeat(tuple(dims_to_be_repeated))

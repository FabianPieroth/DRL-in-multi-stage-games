from typing import Dict, List

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


def batched_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Extends the torch `index_select` function to be used for multiple
    batches at once.

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

    return torch.gather(input, dim, index)


def get_ma_actions(
    learners,
    observations: Dict[int, torch.Tensor],
    deterministic: bool = True,
    excluded_agents: List = None,
):
    if excluded_agents is None:
        excluded_agents = []
    actions = {}
    for agent_id, sa_obs in observations.items():
        if agent_id not in excluded_agents:
            actions[agent_id], _ = learners[agent_id].predict(
                sa_obs, None, episode_start=None, deterministic=deterministic
            )
    return actions

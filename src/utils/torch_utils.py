from typing import Callable, Dict, List, Tuple

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


def torch_inverse_func(
    func: Callable, domain: Tuple[float, float], device="cpu", precision: float = 0.0001
) -> Callable:
    """Compute the numerical inverse of a function and return it as a callable
    that takes batched torch.Tensor arguments. 
    The function is assumed to be strictly monotonic.
    We use a simple piecewise linear approximation.
    NOTE: Provided inputs of inv_func are projected to the function's image

    Args:
        func (Callable): Function to be inverted, is assumed to take torch.Tensor inputs of shape=[batch_size, 1]
        domain (Tuple[float, float]): (lower_bound, upper_bound) for the domain the function should be inverted on
        precision (float, optional): Stepsize to take along domain for linear approximation. Defaults to 0.00001.

    Returns:
        Callable:
    """
    domain_width = domain[1] - domain[0]
    assert domain[1] - domain[0] > 0.0, "The domain needs to be a nonempty interval!"
    num_intervals = round(domain_width / precision)
    domain_boundaries = torch.linspace(
        domain[0], domain[1], num_intervals, device=device
    ).unsqueeze(-1)

    image_boundaries = func(domain_boundaries)

    inv_slopes = get_inv_slopes_of_linear_fun(domain_boundaries, image_boundaries)

    def inv_func(input: torch.Tensor) -> torch.Tensor:
        # TODO: UserWarning: torch.searchsorted(): input value tensor is non-contiguous
        boundary_points = torch.bucketize(
            input[:, 0], image_boundaries[:, 0], right=True
        )
        boundary_points[boundary_points == num_intervals] -= 1
        slope_points = torch.index_select(inv_slopes, dim=0, index=boundary_points)
        intersect_points = torch.index_select(
            domain_boundaries, dim=0, index=boundary_points
        )
        start_points = torch.index_select(
            image_boundaries, dim=0, index=boundary_points
        )
        return intersect_points + slope_points * (input - start_points)

    return inv_func


def get_inv_slopes_of_linear_fun(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Takes two sorted tensors of shape [batch_size, 1], that
    represent (x, y) pairs of a piecewise linear functions.
    It collects the inverse slope via (x_i+1 - x_i)/(y_i+1 - y_i)
    and sets the last entry to 0.0

    Args:
        xs (torch.Tensor): increasing x-values
        ys (torch.Tensor): increasing or decreasing y-values

    Returns:
        torch.Tensor: inverse slopes
    """
    inv_slopes = torch.zeros_like(xs)
    x_diff = xs[1:, :] - xs[:-1, :]
    y_diff = ys[1:, :] - ys[:-1, :]
    inv_slopes[:-1,] = y_diff / x_diff
    return inv_slopes

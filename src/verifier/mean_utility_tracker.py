import torch


class UtilityTracker(object):
    def __init__(self, agent_id: int, device: int, requires_grad: bool = False) -> None:
        self.agent_id = agent_id
        self.device = device
        self.cum_utility = torch.zeros(1, device=device, requires_grad=requires_grad)
        self.cum_sim_count = torch.zeros(1, device=device, requires_grad=requires_grad)

    def add_utility(self, utilities: torch.Tensor):
        """Sum up utilities along batch dimension and add 
        batch size to counter.

        Args:
            utilities (torch.Tensor): shape=[batch_size, 1] or [batch_size]
        """
        if len(utilities.shape) > 1:
            utilities = utilities.squeeze()
        assert (
            len(utilities.shape) == 1
        ), "Expect tensor to have at most 2 dimensions where one is 1 and the other is not!"
        self.cum_sim_count += utilities.shape[0]
        self.cum_utility += utilities.sum()

    def get_mean_utility(self) -> float:
        mean_utility = self.cum_utility / self.cum_sim_count
        return mean_utility.detach().cpu().item()

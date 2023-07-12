from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

from src.utils.torch_utils import batched_index_select


class Mechanism(ABC):
    """
    Auction Mechanism - Interpreted as a Bayesian game. A Mechanism collects
    bids from all players, then allocates available items as well as payments
    for each of the players.
    """

    def play(self, action_profile) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for `run` method"""
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for play for auction mechanisms"""
        raise NotImplementedError()


class FirstPriceAuction(Mechanism):
    """First Price Sealed Bid auction"""

    def __init__(self, random_tie_break: bool = True, **kwargs):
        self.random_tie_break = random_tie_break
        super().__init__(**kwargs)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) First Price Sealed Bid Auction.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert (
            bids.dim() >= 3
        ), "Bid tensor must be at least 3d (*batch_dims x players x items)"

        # TODO can we prevent non-positive bids easily?
        # assert (bids >= 0).all().item(), "All bids must be nonnegative."
        bids[bids < 0] = 0
        # rule_violations = (bids <= 0).any(axis=2)

        device = bids.device

        # name dimensions
        *batch_dims, player_dim, item_dim = range(
            bids.dim()
        )  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        if self.random_tie_break:  # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=bids.device).sort(
                dim=1
            )[1]
            bids = batched_index_select(bids, 1, idx)

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)

        highest_bids, winning_bidders = bids.max(
            dim=player_dim, keepdim=True
        )  # both shapes: [batch_sizes, 1, n_items]

        payments_per_item.scatter_(player_dim, winning_bidders, highest_bids)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)

        # Don't allocate items that have a winning bid of zero.
        allocations.masked_fill_(mask=payments_per_item <= 0, value=0)
        payments.masked_fill_(mask=payments < 0, value=0)

        if self.random_tie_break:  # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            allocations = batched_index_select(allocations, 1, idx_rev)
            payments = batched_index_select(payments, 1, idx_rev)

            # also revert the order of bids if they're used later on
            bids = batched_index_select(bids, 1, idx_rev)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class VickreyAuction(Mechanism):
    "Vickrey / Second Price Sealed Bid Auctions"

    def __init__(self, random_tie_break: bool = True, **kwargs):
        self.random_tie_break = random_tie_break
        super().__init__(**kwargs)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Vickrey/Second Price Sealed Bid Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert (
            bids.dim() >= 3
        ), "Bid tensor must be at least 3d (*batch_dims x players x items)"

        # TODO can we prevent non-positive bids easily?
        # assert (bids >= 0).all().item(), "All bids must be nonnegative."
        bids[bids < 0] = 0

        # name dimensions
        *batch_dims, player_dim, item_dim = range(
            bids.dim()
        )  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        if self.random_tie_break:  # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=bids.device).sort(
                dim=1
            )[1]
            bids = batched_index_select(bids, 1, idx)

        # allocate return variables
        payments_per_item = torch.zeros(
            *batch_sizes, n_players, n_items, device=bids.device
        )
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=bids.device)

        highest_bids, winning_bidders = bids.max(
            dim=player_dim, keepdim=True
        )  # shape of each: [batch_size, 1, n_items]

        # getting the second prices --> price is the lowest of the two highest bids
        top2_bids, _ = bids.topk(2, dim=player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)

        payments_per_item.scatter_(player_dim, winning_bidders, second_prices)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)

        # Don't allocate items that have a winning bid of zero.
        allocations.masked_fill_(mask=payments_per_item <= 0, value=0)
        payments.masked_fill_(mask=payments < 0, value=0)

        if self.random_tie_break:  # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            allocations = batched_index_select(allocations, 1, idx_rev)
            payments = batched_index_select(payments, 1, idx_rev)

            # also revert the order of bids if they're used later on
            bids = batched_index_select(bids, 1, idx_rev)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class AllPayAuction(Mechanism):
    def __init__(
        self, device: Union[str, int], random_tie_break: bool = True, **kwargs
    ):
        self.random_tie_break = random_tie_break
        self.device = device

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) All-Pay Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert bids.dim() >= 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        *batch_dims, player_dim, item_dim = range(bids.dim())
        *batch_sizes, n_players, n_items = bids.shape

        if self.random_tie_break:  # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=bids.device).sort(
                dim=1
            )[1]
            bids = batched_index_select(bids, 1, idx)

        # allocate return variables
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)

        # Assign item to the bidder with the highest bid, in case of a tie assign it to the first one
        _, winning_bidders = bids.max(dim=player_dim, keepdim=True)

        allocations.scatter_(player_dim, winning_bidders, 1)
        allocations.masked_fill_(mask=bids == 0, value=0)

        payments = bids.reshape(*batch_sizes, n_players)  # pay as bid

        if self.random_tie_break:  # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            allocations = batched_index_select(allocations, 1, idx_rev)
            payments = batched_index_select(payments, 1, idx_rev)

            # also revert the order of bids if they're used later on
            bids = batched_index_select(bids, 1, idx_rev)

        return (
            allocations,
            payments,
        )  # payments: batches x players, allocation: batch x players x items


class TullockContest(Mechanism):
    """Tullock Contest"""

    def __init__(
        self, impact_function, device: Union[str, int], use_valuation: bool = True
    ):
        self.device = device

        self.impact_fun = impact_function
        self.use_valuation = use_valuation

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Tullock Contests.

        This function is meant for single-item contests.
        If a effort tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Note that if multiple contestants submit the highest effort, we choose the first one.

        Parameters
        ----------
        bids: torch.Tensor
            of efforts with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert (
            bids.dim() >= 3
        ), "Effort tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All efforts must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(
            bids.dim()
        )  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments = bids.reshape(*batch_sizes, n_players)  # pay as bid

        # transform bids according to impact function
        bids = self.impact_fun(bids)

        # Calculate winning probabilities
        winning_probs = bids / bids.sum(dim=-2, keepdim=True)
        winning_probs[winning_probs.isnan()] = (
            1 / n_players
        )  # equal chances if all contestants exert zero effotr

        if not self.use_valuation:
            winning_probs = winning_probs.reshape(*batch_sizes, n_players, n_items)

        return (
            winning_probs,
            payments,
        )  # payments: batches x players, allocation: batch x players x items

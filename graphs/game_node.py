from typing import Dict

from _types import Coalition, Payoff, Player, Value
from games.coop_game import CoopGame


class GameNode:
    def __init__(self, game: CoopGame, coalition: Coalition, parent: 'GameNode' = None, payoff: Payoff = None):
        self.coalition = coalition
        if payoff is None:
            self.payoff = game.shapely_values(coalition)
        else:
            self.payoff = payoff
        self.value = self.payoff.sum()
        self.children = set()
        self.parents = set()
        if parent is not None:
            self.parents.add(parent)
            parent.children.add(self)

    @property
    def round_payoff(self) -> Payoff:
        return self.payoff.round(2)

    @property
    def payoff_dict(self) -> Dict[Player, Value]:
        round_payoff = self.round_payoff
        return {i: round_payoff[i] for i in self.coalition}

    def loosely_dominates(self, other: 'GameNode') -> bool:
        found_pref = False
        for i in self.coalition:
            if self.round_payoff[i] < other.round_payoff[i]:
                return False  # no one dislikes the situation
            if self.round_payoff[i] > other.round_payoff[i]:
                found_pref = True  # at least one person prefers it
        return found_pref

    def strictly_dominates(self, other: 'GameNode') -> bool:
        for i in self.coalition:
            if self.round_payoff[i] <= other.round_payoff[i]:
                return False  # if someone doesn't prefer this
        return True

    def better_or_equal(self, other: 'GameNode'):
        for i in self.coalition:
            if self.round_payoff[i] < other.round_payoff[i]:
                return False  # no one dislikes the situation
        return True

    def __str__(self):
        return f'GameNode(coalition={self.coalition}, payoff={self.payoff_dict})'

    __repr__ = __str__

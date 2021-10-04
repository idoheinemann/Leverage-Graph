import abc
from _types import Coalition, Value, Player, Payoff
from sympy.utilities.iterables import multiset_permutations
import math
import numpy as np

from tools import normalize_payoff


class CoopGame(abc.ABC):
    def __init__(self, players_amount: int):
        self.grand_coalition = set(range(players_amount))
        self.players_amount = players_amount

    @abc.abstractmethod
    def value(self, coalition: Coalition) -> Value:
        pass

    def added_value(self, coalition: Coalition, player: Player) -> Value:
        return self.value(coalition | {player}) - self.value(coalition - {player})

    def shapely_values(self, coalition: Coalition) -> Payoff:
        payoffs = np.zeros(self.players_amount)
        for perm in multiset_permutations(coalition):
            temp_coalition = set()
            for p in perm:
                payoffs[p] += self.added_value(temp_coalition, p)
                temp_coalition.add(p)
        combs = math.factorial(len(coalition))
        return payoffs / combs

    def shapely_normal(self, coalition: Coalition) -> Payoff:
        return normalize_payoff(self.shapely_values(coalition))

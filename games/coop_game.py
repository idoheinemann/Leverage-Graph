import abc
from _types import Coalition, Value, Player, Payoff
from sympy.utilities.iterables import multiset_permutations
import math

from tools import normalize_payoff


class CoopGame(abc.ABC):
    def __init__(self, players_amount: int):
        self.grand_coalition = set(range(players_amount))

    @abc.abstractmethod
    def value(self, coalition: Coalition) -> Value:
        pass

    def added_value(self, coalition: Coalition, player: Player) -> Value:
        return self.value(coalition | {player}) - self.value(coalition)

    def shapely_values(self, coalition: Coalition) -> Payoff:
        payoffs = {p: 0 for p in coalition}
        for perm in multiset_permutations(coalition):
            temp_coalition = set()
            for p in perm:
                payoffs[p] += self.added_value(temp_coalition, p)
                temp_coalition.add(p)
        combs = math.factorial(len(coalition))
        for i in payoffs:
            payoffs[i] /= combs
        return payoffs

    def shapely_normal(self, coalition: Coalition) -> Payoff:
        return normalize_payoff(self.shapely_values(coalition))

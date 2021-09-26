from _types import Coalition, Value, Payoff
from games.coop_game import CoopGame

import numpy as np


class GlovesGame(CoopGame):
    def __init__(self, players_amount: int, left_amount: int):
        assert 0 <= left_amount <= players_amount
        CoopGame.__init__(self, players_amount)
        self.left = set(range(left_amount))
        self.right = self.grand_coalition - self.left

    def is_winning(self, coalition: Coalition) -> bool:
        return bool(coalition & self.left) and bool(coalition & self.right)

    def value(self, coalition: Coalition) -> Value:
        return float(self.is_winning(coalition))

    def shapely_values(self, coalition: Coalition) -> Payoff:
        left = self.left & coalition
        right = self.right & coalition
        payoff = np.zeros(self.players_amount)
        if len(left) == 0 or len(right) == 0:
            return payoff
        left_opens = len(left) / len(coalition)
        right_opens = len(right) / len(coalition)
        left_value = right_opens / len(left)
        right_value = left_opens / len(right)
        for i in coalition:
            payoff[i] = left_value if i in left else right_value
        return payoff

    def shapely_normal(self, coalition: Coalition) -> Payoff:
        return self.shapely_values(coalition)


if __name__ == '__main__':
    game = GlovesGame(5, 3)
    print(CoopGame.shapely_values(game, game.grand_coalition - {4}))
    print(game.shapely_values(game.grand_coalition - {4}))

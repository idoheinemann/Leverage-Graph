from _types import Coalition, Value, Payoff
from games.coop_game import CoopGame


class ShoesGame(CoopGame):
    def __init__(self, players_amount: int, left_amount: int):
        assert 0 <= left_amount <= players_amount
        CoopGame.__init__(self, players_amount)
        self.left = set(range(left_amount))
        self.right = self.grand_coalition - self.left

    def value(self, coalition: Coalition) -> Value:
        return min(len(coalition & self.left), len(coalition & self.right))


if __name__ == '__main__':
    game = ShoesGame(5, 3)
    print(game.shapely_values(game.grand_coalition))

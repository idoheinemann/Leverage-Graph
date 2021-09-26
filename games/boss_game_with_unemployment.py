from _types import Coalition, Value, Payoff
from games.coop_game import CoopGame


class BossGameWithUnemployment(CoopGame):
    def __init__(self, players_amount: int):
        CoopGame.__init__(self, players_amount + 1)

    def value(self, coalition: Coalition) -> Value:
        if 0 in coalition:
            return len(coalition) - 1
        return len(coalition) / 8


if __name__ == '__main__':
    game = BossGameWithUnemployment(5)
    print(game.shapely_values(game.grand_coalition))

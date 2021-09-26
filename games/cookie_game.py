from _types import Coalition, Value, Payoff
from games.coop_game import CoopGame


class CookieGame(CoopGame):
    def __init__(self, *args, value_factor=1.1):
        CoopGame.__init__(self, len(args))
        self.values = args
        self.value_factor = value_factor

    def value(self, coalition: Coalition) -> Value:
        return round(sum(self.values[x] for x in coalition) * self.value_factor ** len(coalition))


if __name__ == '__main__':
    game = CookieGame(5, 3, 7, 1)
    print(game.shapely_values(game.grand_coalition))

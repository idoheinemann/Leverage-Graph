from _types import Coalition, Value, Payoff
from games.coop_game import CoopGame


class PipelineGame(CoopGame):
    def __init__(self, *args):
        CoopGame.__init__(self, sum(x for x in args))
        self.pipes = [set() for _ in args]
        current_index = 0
        acc = args[current_index]
        for i in range(len(self.grand_coalition)):
            if acc <= i:
                current_index += 1
                acc += args[current_index]
            self.pipes[current_index].add(i)

    def value(self, coalition: Coalition) -> Value:
        for i in self.pipes:
            if not (i & coalition):
                return 0
        return 1


if __name__ == '__main__':
    game = PipelineGame(2, 1, 1)
    print(game.shapely_values(game.grand_coalition))

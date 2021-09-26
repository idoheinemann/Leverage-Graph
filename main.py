from games.boss_game_with_unemployment import BossGameWithUnemployment
from games.gloves_game import GlovesGame
from games.pipeline_game import PipelineGame
from graphs.leverage_graph import LeverageGraph


def main():
    game = PipelineGame(4, 2, 3)
    tree = LeverageGraph(game, leverage_epsilon=0.01)
    print(tree.strictly_dominant_set())
    lev = tree.evaluate_leverage()
    print(lev.strictly_dominant_set())
    stable, i = tree.find_stable()
    print(i)
    print(stable.strictly_dominant_set())
    trees, i, j = tree.find_stable_circulation()
    print((i, j))
    avg = tree.average_tree(10)
    print(avg.strictly_dominant_set())


if __name__ == '__main__':
    main()

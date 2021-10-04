import itertools
from typing import Dict, Tuple, Set, List

from _types import Player, Value, Payoff
from games.coop_game import CoopGame
from graphs.game_graph import GameGraph, GameNode
from tools import normalize_payoff

import numpy as np


class LeverageGraph(GameGraph):
    def __init__(self, game: CoopGame, *args, leverage_epsilon: Value = 0, **kwargs):
        GameGraph.__init__(self, game, *args, **kwargs)
        self.leverage_epsilon = leverage_epsilon

    def threat_states(self, state: GameNode, player: Player, opponent: Player) -> Set[Tuple[GameNode, Value]]:
        all_with_player = {x for x in self.to_set() if
                           player in x.coalition and x.payoff[player] >= state.payoff[player]}
        all_where_credible: Set[GameNode] = set()
        for i in all_with_player:
            sum_of_losses = sum(
                max(state.payoff[x] - i.payoff[x], 0) for x in i.coalition & state.coalition - {opponent})
            # sum of losses of all players who's agreement is needed in order to switch from state to i
            if sum_of_losses <= i.payoff[player] - state.payoff[player]:
                # if player can make up the losses of all players
                all_where_credible.add(i)

        all_where_credible.remove(state)
        all_without_player = {x for x in self.to_set() if opponent in x.coalition and player not in x.coalition}
        filtered_credible = set()
        for i in all_where_credible:
            # how much the player can demand from the opponent
            # by telling him that he'll switch from state to i if he does not comply
            threat = state.payoff[opponent] - i.payoff[opponent] - self.leverage_epsilon
            if threat <= 0:
                continue
            init_treat = threat
            for j in all_without_player:
                player_marginal = i.payoff[player] - state.payoff[player]
                # sum of how much each player prefers i to j
                # and how much player can add to them while remaining profitable
                # in order to counter any offer from opponent
                if i.coalition & j.coalition:
                    # if there is any overlap between the coalitions, it is enough for player to convince the
                    # overlapping players not to switch from i to j
                    accumulated = sum(i.payoff[x] - j.payoff[x] for x in j.coalition & i.coalition) + \
                                  player_marginal

                else:
                    # if there is no overlap, player must pay the difference to all players in j in order
                    # to prevent offers from opponent
                    accumulated = player_marginal - j.value + i.payoff[opponent]
                    if accumulated < 0:
                        # player cant prevent moving to j
                        accumulated = 0
                # for every coalition with opponent and without player
                # subtract how much other players in both coalitions prefer this option (plus how much player can add)
                # from how much does the opponent gain from j relative to i (spare opponent gains)
                # if opponent can pay all players in the coalition the difference and still come out on top
                # then player must subtract what remains from the threat in order for it to still pay off for opponent

                op_remain = j.payoff[opponent] - i.payoff[opponent] - accumulated
                threat = min(threat, init_treat - op_remain)
                if threat <= 0:
                    break
            else:
                filtered_credible.add((i, threat))

        return filtered_credible

    def leverages(self, state: GameNode, player: Player, opponent: Player, threat_states=None) -> List[Value]:
        if threat_states is None:
            threat_states = self.threat_states(state, player, opponent)
        if len(threat_states) == 0:
            return []  # this is the best state for player, no leverage at all
        return [x[1] for x in sorted(threat_states, key=lambda x: x[1])]

    def get_leverage_vector(self, state: GameNode) -> Dict[Player, Payoff]:
        leverages = {p: np.zeros(self.game.players_amount) for p in state.coalition}
        for p1, p2 in itertools.combinations(state.coalition, 2):
            p1_leverage = self.leverages(state, p1, p2)
            p2_leverage = self.leverages(state, p2, p1)
            if len(p1_leverage) == 0 and len(p2_leverage) == 0:
                leverages[p2][p1] = 0
                leverages[p1][p2] = 0
            elif len(p1_leverage) == 0:
                leverages[p2][p1] = 0
                leverages[p1][p2] = p2_leverage[0]
            elif len(p2_leverage) == 0:
                leverages[p2][p1] = p1_leverage[0]
                leverages[p1][p2] = 0
            else:
                dlp1, dlp2 = self.handle_double_leverage(p1_leverage, p2_leverage)
                leverages[p2][p1] = dlp1
                leverages[p1][p2] = dlp2
        # if both of them have leverage on the other
        # still not sure if this is the best way to calculate this leverage

        return leverages

    @staticmethod
    def handle_double_leverage(p1_leverage, p2_leverage) -> Tuple[Value, Value]:
        return max(p1_leverage[0] - p2_leverage[0], 0), max(p2_leverage[0] - p1_leverage[0], 0)

    def evaluate_leverage(self) -> 'LeverageGraph':
        new_tree = self.deepcopy()
        for state in self.to_set():
            if len(state.coalition) == 1:
                continue
            state_copy = new_tree.search(state.coalition)
            leverage_vector = self.get_leverage_vector(state)
            for player in leverage_vector:
                max_pay = max(leverage_vector[player])
                normal = normalize_payoff(leverage_vector[player])
                state_copy.payoff += normal * max_pay
                state_copy.payoff[player] -= max_pay
        return new_tree

    def find_stable(self, max_iter=10) -> Tuple['LeverageGraph', int]:
        current_tree = self
        for i in range(max_iter):
            new_tree = current_tree.evaluate_leverage()
            if new_tree == current_tree:
                return new_tree, i
            current_tree = new_tree
        return current_tree, -1

    def find_stable_circulation(self, max_iter=10) -> Tuple[List['LeverageGraph'], int, int]:
        current_tree = self.deepcopy()
        trees = [current_tree]
        for i in range(max_iter):
            current_tree = current_tree.evaluate_leverage()
            for j, t in enumerate(trees):
                if t == current_tree:
                    return trees, i, j
            trees.append(current_tree)
        return trees, -1, -1

    def average_stable_tree(self, max_iter=10) -> 'LeverageGraph':
        trees, _, _ = self.find_stable_circulation(max_iter)
        base_tree = trees.pop()
        for i in trees:
            for j in i.to_set():
                base_tree.search(j.coalition).payoff += j.payoff
        for i in base_tree.to_set():
            i.payoff /= len(trees) + 1
        return base_tree

    def average_tree(self, iterations=10) -> 'LeverageGraph':
        trees = []
        current_tree = self
        for i in range(iterations):
            current_tree = current_tree.evaluate_leverage()
            trees.append(current_tree)
        base_tree = self.deepcopy()
        for i in trees:
            for j in i.to_set():
                base_tree.search(j.coalition).payoff += j.payoff
        for i in base_tree.to_set():
            i.payoff /= len(trees) + 1
        return base_tree

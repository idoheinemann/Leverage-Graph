import itertools
from typing import Dict, Tuple, Set, List

from _types import Player, Value, Payoff
from games.coop_game import CoopGame
from graphs.game_graph import GameGraph, GameNode
from tools import normalize_payoff

import numpy as np


class LeverageGraph(GameGraph):
    def __init__(self, game: CoopGame, *args, leverage_epsilon: Value = 0,
                 threat_help_epsilon: Value = 0, threat_enable_epsilon: Value = 0, complete_missing_states=False,
                 **kwargs):
        GameGraph.__init__(self, game, *args, **kwargs)
        self.leverage_epsilon = leverage_epsilon
        self.threat_help_epsilon = threat_help_epsilon
        self.threat_enable_epsilon = threat_enable_epsilon
        if complete_missing_states:
            root = self.root
            while len(root.parents) != 0:
                root = next(iter(root.parents))
            self.root = root
            self._recursive_make_nodes_down(self.root)
            self._merge_same_coalition()
            self._game_set = None

    def sum_of_losses(self, state: GameNode, new_state: GameNode, player: Player, opponent: Player) -> Value:
        """
        the collective losses for players from joining the new coalition (except for opponent)
        if the player wants to move to the new state, he will need to somehow make up for that collective loss
        :param state:
        :param new_state:
        :param player:
        :param opponent:
        :return:
        """
        return sum(max(state.payoff[x] - new_state.payoff[x] + (self.threat_enable_epsilon if player != x else 0), 0) for x in
                   new_state.coalition & state.coalition - {opponent})

    def sum_of_gains(self, state: GameNode, new_state: GameNode, player: Player, opponent: Player) -> Value:
        """
        the collective gain every player in the new coalition (except opponent) can give to make up for
        accumulated losses to other players
        :param state:
        :param new_state:
        :param player:
        :param opponent:
        :return:
        """
        return sum(
            max(new_state.payoff[x] - state.payoff[x] - (self.threat_help_epsilon if x != player else 0), 0) for x
            in new_state.coalition - {opponent})

    def filter_incredible_states(self, state: GameNode, player: Player, opponent: Player,
                                 threat_states: Set[GameNode]) -> Set[GameNode]:
        all_where_credible: Set[GameNode] = set()
        for new_state in threat_states:
            if state is new_state:
                continue
            sum_of_losses = self.sum_of_losses(state, new_state, player, opponent)
            # all losses accumulated by all players except opponent
            sum_of_gains = self.sum_of_gains(state, new_state, player, opponent)
            # all gains accumulated by all players except opponent

            # sum of losses of all players who's agreement is needed in order to switch from state to new_state
            # must be less then the sum of gains of all players who's agreement is necessary
            # because that would mean that the gaining players can make up for the losses and still come out on top
            if sum_of_losses <= sum_of_gains:
                # if player can make up the losses of all players
                all_where_credible.add(new_state)

        return all_where_credible

    def credible_passable_states(self, state: GameNode, player: Player, opponent: Player) -> Set[GameNode]:
        """
        finds all states to which player can threaten opponent to switch to
        regardless of whether opponent can block that threat with a counter threat
        :param state:
        :param player:
        :param opponent:
        :return:
        """
        all_with_player = {x for x in self.to_set() if
                           player in x.coalition and x.payoff[player] >= state.payoff[player]}
        return self.filter_incredible_states(state, player, opponent, all_with_player)

    def threat_states(self, state: GameNode, player: Player, opponent: Player) -> Set[Tuple[GameNode, Value]]:
        all_where_credible = self.credible_passable_states(state, player, opponent)
        all_without_player = {x for x in self.to_set() if opponent in x.coalition and player not in x.coalition}
        filtered_credible = set()
        for i in all_where_credible:
            # how much the player can demand from the opponent
            # by telling him that he'll switch from state to i if he does not comply
            init_treat = state.payoff[opponent] - i.payoff[opponent] - self.leverage_epsilon
            if init_treat <= 0:
                continue
            threat = init_treat
            for j in self.filter_incredible_states(i, opponent, player, all_without_player):
                accumulated = self.sum_of_losses(i, j, opponent, player)
                opponent_margin = max(j.payoff[opponent] - i.payoff[opponent] - accumulated, 0)
                # how much opponent has after compensating all losses, assuming he compensates first
                # because he demanded the move
                # (sometimes he will not be able to compensate alone, so in that case his marginal profit is 0)
                threat = min(threat, init_treat - opponent_margin)
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
            i.payoff /= iterations + 1
        return base_tree

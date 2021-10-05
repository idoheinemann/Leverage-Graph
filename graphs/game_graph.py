import itertools
from typing import Optional, Set

from _types import Coalition, Payoff
from games.coop_game import CoopGame
from tools import normalize_payoff
from graphs.game_node import GameNode

import numpy as np


class GameGraph:
    def __init__(self, game: CoopGame, coalition: Optional[Coalition] = None, payoff: Optional[Payoff] = None,
                 merge_same_coalition=True, set_non_profitable_to_zero=False):
        if coalition is None:
            coalition = game.grand_coalition
        self.game = game
        self.set_non_profitable_to_zero = set_non_profitable_to_zero
        self.root = GameNode(game, coalition, payoff=payoff)
        self._recursive_make_nodes(self.root)
        self._game_set = None
        if merge_same_coalition:
            self._merge_same_coalition()
        self._game_set = None

    def _merge_same_coalition(self):
        game_list = list(self.to_set())
        merge_dict = {}
        i = 0
        while i < len(game_list) - 1:
            j = i + 1
            merge_dict[game_list[i]] = []
            while j < len(game_list):
                if game_list[i].coalition == game_list[j].coalition:
                    merge_dict[game_list[i]].append(game_list.pop(j))
                else:
                    j += 1
            i += 1
        for first in merge_dict:
            for k in merge_dict[first]:
                first.payoff += k.payoff
                first.parents |= k.parents
                first.children |= k.children
                for parent in k.parents:
                    parent.children.remove(k)
                    parent.children.add(first)
                for child in k.children:
                    child.parents.remove(k)
                    child.parents.add(first)
            first.payoff /= len(merge_dict[first]) + 1

    def search(self, coalition: Coalition, start_node: GameNode = None) -> Optional[GameNode]:
        return self.search_down(coalition, start_node) or self.search_up(coalition, start_node)

    def search_down(self, coalition: Coalition, start_node: GameNode = None) -> Optional[GameNode]:
        if start_node is None:
            start_node = self.root
        if start_node.coalition == coalition:
            return start_node
        for i in start_node.children:
            if len(coalition - i.coalition) == 0:
                # if some element belongs to the coalition but not to i no child of i can represent the coalition
                val = self.search_down(coalition, i)
                if val is not None:
                    return val
        return None

    def search_up(self, coalition: Coalition, start_node: GameNode = None) -> Optional[GameNode]:
        if start_node is None:
            start_node = self.root
        if start_node.coalition == coalition:
            return start_node
        for i in start_node.parents:
            if len(i.coalition - coalition) == 0:
                # if some element belongs to i but not to the coalition no parent of i can represent the coalition
                val = self.search_up(coalition, i)
                if val is not None:
                    return val
        return None

    def _find_all_nodes(self, coalition: Coalition, start_node: GameNode = None) -> Set[GameNode]:
        return self._find_all_nodes_down(coalition, start_node) | \
               self._find_all_nodes_up(coalition, start_node)

    def _find_all_nodes_up(self, coalition: Coalition, start_node: GameNode = None, found_set=None) -> Set[GameNode]:
        if found_set is None:
            found_set = set()
        if start_node is None:
            start_node = self.root
        if start_node.coalition == coalition:
            found_set.add(start_node)
        for i in start_node.parents:
            if len(i.coalition - coalition) == 0:
                # if some element belongs to the coalition but not to i no child of i can represent the coalition
                self._find_all_nodes_up(coalition, i, found_set=found_set)
        return found_set

    def _find_all_nodes_down(self, coalition: Coalition, start_node: GameNode = None, found_set=None) -> Set[GameNode]:
        if found_set is None:
            found_set = set()
        if start_node is None:
            start_node = self.root
        if start_node.coalition == coalition:
            found_set.add(start_node)
        for i in start_node.children:
            if len(coalition - i.coalition) == 0:
                # if some element belongs to the coalition but not to i no child of i can represent the coalition
                self._find_all_nodes_down(coalition, i, found_set=found_set)
        return found_set

    def _recursive_make_nodes(self, node: GameNode):
        self._recursive_make_nodes_up(node)
        self._recursive_make_nodes_down(node)

    def _recursive_make_nodes_up(self, node: GameNode):
        for player in self.game.grand_coalition - node.coalition:
            super_coalition = node.coalition | {player}
            if self.search_up(super_coalition, node):
                continue  # coalition already exists
            payoff = self.game.shapely_values(super_coalition)
            value = payoff.sum()
            diff = value - node.value
            if diff < 0:
                if self.set_non_profitable_to_zero:
                    payoff = np.zeros(payoff.shape)
            else:
                added_value = np.maximum(payoff - node.payoff, 0)
                # each player's added gain from joining the new player is his contribution
                # to making the new value
                payoff = node.payoff + (normalize_payoff(added_value) * diff)
            new_node = GameNode(self.game, super_coalition, payoff=payoff)
            new_node.children.add(node)
            node.parents.add(new_node)
            self._recursive_make_nodes_up(new_node)

    def _recursive_make_nodes_down(self, node: GameNode):
        if len(node.coalition) == 1:
            return
        for player in node.coalition:
            sub_coalition = node.coalition - {player}
            if self.search_down(sub_coalition, node):
                continue  # coalition already exists
            parents_set = self.to_set_up(node)
            payoff = self.game.shapely_values(sub_coalition)
            value = payoff.sum()
            current = node
            for i in parents_set:
                if i.loosely_dominates(current):
                    current = i
            # current is now the best upper state
            non_existing = current.coalition - sub_coalition
            diff = value - (current.value - sum(current.payoff[x] for x in non_existing))
            if diff >= 0:
                normal = normalize_payoff(payoff)
                payoff = current.payoff + (normal * diff)
                for i in non_existing:
                    payoff[i] = 0
            else:
                if self.set_non_profitable_to_zero:
                    payoff = np.zeros(payoff.shape)
            new_node = GameNode(self.game, sub_coalition, node, payoff=payoff)
            self._recursive_make_nodes_down(new_node)

    def to_set(self) -> Set[GameNode]:
        if self._game_set is None:
            self._game_set = self.to_set_down() | self.to_set_up()
        return self._game_set

    def to_set_up(self, node: GameNode = None, existing: Optional[Set[GameNode]] = None) -> Set[GameNode]:
        if node is None:
            node = self.root
        if existing is None:
            existing = set()
        for i in node.parents:
            if i not in existing:
                existing.add(i)
                self.to_set_up(i, existing)
        return existing

    def to_set_down(self, node: GameNode = None, existing: Optional[Set[GameNode]] = None) -> Set[GameNode]:
        if node is None:
            node = self.root
        if existing is None:
            existing = {node}
        for i in node.children:
            if i not in existing:
                existing.add(i)
                self.to_set_down(i, existing)
        return existing

    def strictly_dominant_set(self) -> Set[GameNode]:
        states = self.to_set().copy()
        for i, j in itertools.combinations(self.to_set(), 2):
            if not (i.coalition & j.coalition):
                continue
            if i.strictly_dominates(j) and j in states:
                states.remove(j)
            if j.strictly_dominates(i) and i in states:
                states.remove(i)
        return states

    def loosely_dominant_set(self) -> Set[GameNode]:
        states = self.to_set().copy()
        for i, j in itertools.combinations(self.to_set(), 2):
            if not (i.coalition & j.coalition):
                continue
            if i.loosely_dominates(j) and j in states:
                states.remove(j)
            if j.loosely_dominates(i) and i in states:
                states.remove(i)
        return states

    def better_or_equal_set(self) -> Set[GameNode]:
        states = self.to_set().copy()
        for i, j in itertools.combinations(self.to_set(), 2):
            if not (i.coalition & j.coalition):
                continue
            if i.better_or_equal(j) and j in states:
                states.remove(j)
            if j.better_or_equal(i) and i in states:
                states.remove(i)
        return states

    def deepcopy(self) -> 'GameGraph':
        from copy import deepcopy
        return deepcopy(self)

    def all_strictly_dominant(self, node: GameNode) -> Set[GameNode]:
        return {i for i in self.to_set() if i.strictly_dominates(node)}

    def all_loosely_dominant(self, node: GameNode) -> Set[GameNode]:
        return {i for i in self.to_set() if i.loosely_dominates(node)}

    def all_better_or_equal(self, node: GameNode) -> Set[GameNode]:
        return {i for i in self.to_set() if i.better_or_equal(node)}

    def __eq__(self, other: 'GameGraph') -> bool:
        if len(other.to_set()) != len(self.to_set()):
            return False
        for state in self.to_set():
            other_state = other.search(state.coalition)
            if other_state is None:
                return False
            if other_state.payoff_dict != state.payoff_dict:
                return False
        return True


def main():
    from games.gloves_game import GlovesGame
    tree = GameGraph(GlovesGame(5, 3))
    print(tree)


if __name__ == '__main__':
    main()

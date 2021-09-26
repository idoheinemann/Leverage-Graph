import itertools
from typing import Optional, Set, Dict

from _types import Coalition, Player, Value, Payoff
from games.coop_game import CoopGame
from tools import normalize_payoff


class GameNode:
    def __init__(self, game: CoopGame, coalition: Coalition, parent: 'GameNode' = None, payoff: Payoff = None):
        self.coalition = coalition
        if payoff is None:
            self.payoff = game.shapely_values(coalition)
        else:
            self.payoff = payoff
        self.value = self.payoff.sum()
        self.children = set()
        self.parents = set()
        if parent is not None:
            self.parents.add(parent)
            parent.children.add(self)

    @property
    def round_payoff(self) -> Payoff:
        return self.payoff.round(2)

    @property
    def payoff_dict(self) -> Dict[Player, Value]:
        round_payoff = self.round_payoff
        return {i: round_payoff[i] for i in self.coalition}

    def loosely_dominates(self, other: 'GameNode') -> bool:
        found_pref = False
        for i in self.coalition:
            if self.round_payoff[i] < other.round_payoff[i]:
                return False  # no one dislikes the situation
            if self.round_payoff[i] > other.round_payoff[i]:
                found_pref = True  # at least one person prefers it
        return found_pref

    def strictly_dominates(self, other: 'GameNode') -> bool:
        for i in self.coalition:
            if self.round_payoff[i] <= other.round_payoff[i]:
                return False  # if someone doesn't prefer this
        return True

    def better_or_equal(self, other: 'GameNode'):
        for i in self.coalition:
            if self.round_payoff[i] < other.round_payoff[i]:
                return False  # no one dislikes the situation
        return True

    def __str__(self):
        return f'GameNode(coalition={self.coalition}, payoff={self.payoff_dict})'

    __repr__ = __str__


class GameGraph:
    def __init__(self, game: CoopGame, coalition: Optional[Coalition] = None, merge_same_coalition=True):
        if coalition is None:
            coalition = game.grand_coalition
        self.game = game
        self.coalition = coalition
        self.root = GameNode(game, coalition)
        self._recursive_make_nodes(self.root)
        if merge_same_coalition:
            for i in range(1, len(coalition)):
                for j in itertools.combinations(coalition, i):
                    all_nodes = self._find_all_nodes(set(j))
                    iter_nodes = iter(all_nodes)
                    first = next(iter_nodes)
                    for k in iter_nodes:
                        first.payoff += k.payoff
                        first.parents |= k.parents
                        for parent in k.parents:
                            parent.children.remove(k)
                            parent.children.add(first)
                    first.payoff /= len(all_nodes)
        self.game_set = None

    def search_down(self, coalition: Coalition, start_node: Optional[GameNode] = None) -> Optional[GameNode]:
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

    def search_up(self, coalition: Coalition, start_node: GameNode) -> Optional[GameNode]:
        if start_node.coalition == coalition:
            return start_node
        for i in start_node.parents:
            if len(i.coalition - coalition) != 0:
                # if some element belongs to i but not to the coalition no parent of i can represent the coalition
                val = self.search_up(coalition, i)
                if val is not None:
                    return val
        return None

    def _find_all_nodes(self, coalition: Coalition, start_node: GameNode = None, found_set=None) -> Set[GameNode]:
        if found_set is None:
            found_set = set()
        if start_node is None:
            start_node = self.root
        if start_node.coalition == coalition:
            found_set.add(start_node)
        for i in start_node.children:
            if len(coalition - i.coalition) == 0:
                # if some element belongs to the coalition but not to i no child of i can represent the coalition
                self._find_all_nodes(coalition, i, found_set=found_set)
        return found_set

    def _recursive_make_nodes(self, node: GameNode):
        if len(node.coalition) == 1:
            return
        for player in node.coalition:
            sub_coalition = node.coalition - {player}
            parents_set = self._to_set_up(node)
            payoff = self.game.shapely_values(sub_coalition)
            value = payoff.sum()
            current = node
            for i in parents_set:
                if i.loosely_dominates(current):
                    current = i
            non_existing = current.coalition - sub_coalition
            diff = value - current.value + sum(current.payoff[x] for x in non_existing)
            # current is now the best upper state
            if diff >= 0:
                normal = normalize_payoff(payoff)
                payoff = current.payoff.copy()
                payoff += normal * diff
                for i in non_existing:
                    payoff[i] = 0
            else:
                pass
                # not everyone have incentive to remove player
                # payoff = np.zeros(self.game.players_amount)
            new_node = GameNode(self.game, sub_coalition, node, payoff=payoff)
            self._recursive_make_nodes(new_node)

    def to_set(self) -> Set[GameNode]:
        if self.game_set is None:
            self.game_set = self._to_set_down()
        return self.game_set

    def _to_set_down(self, node: Optional[GameNode] = None, existing: Optional[Set[GameNode]] = None) -> Set[GameNode]:
        if node is None:
            node = self.root
        if existing is None:
            existing = {node}
        for i in node.children:
            if i not in existing:
                existing.add(i)
                self._to_set_down(i, existing)
        return existing

    def _to_set_up(self, node: GameNode, existing: Optional[Set[GameNode]] = None) -> Set[GameNode]:
        if existing is None:
            existing = set()
        for i in node.parents:
            if i not in existing:
                existing.add(i)
                self._to_set_up(i, existing)
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

    def deepcopy(self) -> 'GameGraph':
        from copy import deepcopy
        return deepcopy(self)

    def all_strictly_dominant(self, node: GameNode):
        return {i for i in self.to_set() if i.strictly_dominates(node)}

    def __eq__(self, other: 'GameGraph') -> bool:
        if len(other.to_set()) != len(self.to_set()):
            return False
        for state in self.to_set():
            other_state = other.search_down(state.coalition)
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

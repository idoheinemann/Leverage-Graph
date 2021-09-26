from _types import Payoff


def normalize_payoff(payoff: Payoff) -> Payoff:
    s = payoff.sum()
    if s == 0:
        return payoff
    return payoff / s

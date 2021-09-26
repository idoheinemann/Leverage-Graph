from _types import Payoff, Value


def normalize_payoff(payoff: Payoff) -> Payoff:
    payoff_sum = sum(payoff.values())
    if payoff_sum == 0:
        return payoff
    for i in payoff:
        payoff[i] /= payoff_sum
    return payoff


def add_payoff(pay1: Payoff, pay2: Payoff) -> Payoff:
    for x in pay1:
        pay1[x] += pay2[x]
    return pay1


def mul_payoff(payoff: Payoff, scalar: Value) -> Payoff:
    for x in payoff:
        payoff[x] *= scalar
    return payoff

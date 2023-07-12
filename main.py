"""
This file implements Formula 6 (Expected value of our holdings at end of our investment horizon)
and compares different choices of parameters
    - sigma: idiosyncratic risk (currently held at 20%)
    - t_h: expected lifetime of investment (currently held at 5 years)
    - ror: current manager rate of return (currently held at 6%)
    - r_prime: alternative manager rate of return
    - p_naught: current fund's value at time t=0
    - p_tau: value of the high water NAV
    - f: fee payment (currently held at 20%)
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def calculate_expected_value(sigma: float, t_h: int, ror: float, p_naught: float, p_tau: float, fee: float) -> float:
    """ Calculate expected value of holdings at time t = t_h. Parameters defined in documentation.
    """
    nav = p_naught * np.exp(ror * t_h)
    fee_payment = calculate_fees(sigma, t_h, ror, p_naught, p_tau, fee)

    expected_value = nav - fee_payment
    return expected_value


def calculate_fees(sigma, t_h, ror, p_naught, p_tau, fee):
    N = norm.cdf  # define a normal distribution

    d_pos_numerator = np.log(p_naught / p_tau) + t_h * (ror + sigma ** 2 / 2)
    d_pos = d_pos_numerator / (sigma * np.sqrt(t_h))

    d_neg_numerator = np.log(p_naught / p_tau) + t_h * (ror - sigma ** 2 / 2)
    d_neg = d_neg_numerator / (sigma * np.sqrt(t_h))

    fee_payment = fee * (N(d_pos) * p_naught * np.exp(ror * t_h) - N(d_neg) * p_tau)
    return max(fee_payment, 0.0)


if __name__ == '__main__':
    sigma = 0.20
    t_h = 5
    r = 0.06
    p_naught = 100
    fee = 0.20

    p_tau = 180
    ror_primes = np.linspace(start=0.06, stop=0.16, num=20)  # a range of r'

    alternative_values = [calculate_expected_value(sigma, t_h, ror, p_naught, p_naught, fee) for ror in ror_primes]
    current_manager_value = calculate_expected_value(sigma, t_h, r, p_naught, p_tau, fee)

    plt.plot(ror_primes, alternative_values, label='alternative manager expected value')
    plt.axhline(current_manager_value, color='r', linestyle='-', label='current manager expected value')
    plt.xlabel('rors for alternative manager')
    plt.ylabel('expected value')
    plt.legend()
    plt.show()

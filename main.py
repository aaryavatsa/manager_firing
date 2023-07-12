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


def calculate_fees(sigma: float, t_h: int, ror: float, p_naught: float, p_tau: float, fee: float) -> float:
    N = norm.cdf  # define a normal distribution

    d_pos_numerator = np.log(p_naught / p_tau) + t_h * (ror + sigma ** 2 / 2)
    d_pos = d_pos_numerator / (sigma * np.sqrt(t_h))

    d_neg_numerator = np.log(p_naught / p_tau) + t_h * (ror - sigma ** 2 / 2)
    d_neg = d_neg_numerator / (sigma * np.sqrt(t_h))

    fee_payment = fee * (N(d_pos) * p_naught * np.exp(ror * t_h) - N(d_neg) * p_tau)
    return max(fee_payment, 0.0)


def numerical_simulation(num_simulations: int, sigma: float, delta_t: float,
                         ror: float, p_naught: float, p_tau: float, fee: float) -> float:
    fees = []
    terminal_values = []
    nav_naught = p_naught * np.exp(ror * delta_t)
    total_crystallization_periods = int(5 / delta_t)  # holding t_h at 5 years... can make this another variable

    for _ in range(num_simulations):  # number of paths
        curr_high_watermark = nav_naught
        curr_nav = nav_naught
        for _ in range(1, total_crystallization_periods):  # iterations for one path
            normal_rv = np.random.normal()
            exponential_term = delta_t * (ror - sigma ** 2 / 2) + (normal_rv * sigma * np.sqrt(delta_t))
            new_nav = curr_nav * np.exp(exponential_term)
            if new_nav > curr_high_watermark:
                curr_high_watermark = new_nav
            curr_nav = new_nav

        path_fee = fee * max(curr_high_watermark - p_tau, 0.0)
        fees.append(path_fee)
        terminal_values.append(curr_nav)

    average_fee = sum(fees) / len(fees)
    avg_terminal_value = sum(terminal_values) / len(terminal_values)
    avg_expected_value = avg_terminal_value - average_fee
    # print(f'Average expected value: {avg_expected_value}')
    return avg_expected_value


if __name__ == '__main__':
    sigma = .400
    t_h = 1
    r = 0.05
    p_naught = 100
    fee = 0.20

    p_tau1 = 120
    p_tau2 = 180
    ror_primes = np.linspace(start=r, stop=0.10, num=20)  # a range of r'

    alternative_values = [calculate_expected_value(sigma, t_h, ror, p_naught, p_naught, fee) for ror in ror_primes]
    cm1 = calculate_expected_value(sigma, t_h, r, p_naught, p_tau1, fee)
    cm2 = calculate_expected_value(sigma, t_h, r, p_naught, p_tau2, fee)
#    print(cm2-cm1)
    current_manager_values1 = [cm1 for ror in ror_primes]
    current_manager_values2 = [cm2 for ror in ror_primes]

    plt.title(r'Comparison for $T_H=%i$' %t_h + ', $\sigma=%1.1f$' %sigma)
    plt.plot(ror_primes, alternative_values, label='alternative manager')
    plt.plot(ror_primes, current_manager_values1, color ='r',label=r'current manager, $P_\tau=120$')
    plt.plot(ror_primes, current_manager_values2, color ='g',label=r'current manager, $P_\tau=180$')
#    plt.axhline(current_manager_value, color='r', linestyle='-', label='current manager expected value')
    plt.xlabel(r'$r^\prime$')
    plt.ylabel('Expected value')
    plt.legend()
    plt.show()

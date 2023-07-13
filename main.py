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
                         ror: float, p_naught: float, fee: float, t_h: float) -> tuple[float, float]:
    fees = []
    terminal_values = []
    num_crystallization_periods = int(t_h / delta_t)

    for _ in range(num_simulations):  # number of paths
        curr_high_watermark = p_naught
        curr_p = p_naught
        for _ in range(num_crystallization_periods):  # iterations for one path
            normal_rv = np.random.normal()
            exponential_term = delta_t * (ror - sigma ** 2 / 2) + (normal_rv * sigma * np.sqrt(delta_t))
            p_i = curr_p * np.exp(exponential_term)
            if p_i > curr_high_watermark:
                curr_high_watermark = p_i
            curr_p = p_i

        path_fee = fee * max(curr_high_watermark - p_naught, 0.0)
        fees.append(path_fee)
        terminal_values.append(curr_p)

    expected_values = [terminal_values[i] - fees[i] for i in range(num_simulations)]
    avg_expected_value = np.average(expected_values)
    std_error = np.std(expected_values, ddof=1) / np.sqrt(num_simulations)

    # print(f'avg value: {avg_expected_value}, std_err: {std_error}')
    return avg_expected_value, std_error


if __name__ == '__main__':
    sigma = .200
    t_h = 5
    r = 0.05
    p_naught = 100
    fee = 0.20

    p_tau1 = 120
    p_tau2 = 180
    delta_t = 0.25
    ror_primes = np.linspace(start=r, stop=0.10, num=20)  # a range of r'

    alt_values = [calculate_expected_value(sigma, t_h, ror, p_naught, p_naught, fee) for ror in ror_primes]

    sim_values = []
    sim_std_err = []
    for i in range(len(ror_primes)):
        simulation = numerical_simulation(100000, sigma, delta_t, ror_primes[i], p_naught, fee, t_h)
        sim_values.append(simulation[0])
        sim_std_err.append(simulation[1])
        print(f'simulation {i + 1}/{len(ror_primes)} complete.')

    cm1 = calculate_expected_value(sigma, t_h, r, p_naught, p_tau1, fee)
    cm2 = calculate_expected_value(sigma, t_h, r, p_naught, p_tau2, fee)
    #    print(cm2-cm1)
    current_manager_values1 = [cm1 for ror in ror_primes]
    current_manager_values2 = [cm2 for ror in ror_primes]

    plt.xlabel(r'$r^\prime$')
    plt.ylabel('Expected value')
    plt.title(r'Comparison for $T_H=%i$' % t_h + ', $\sigma=%1.1f$' % sigma)

    plt.plot(ror_primes, alt_values, label='alternative manager, approach 1')
    plt.errorbar(ror_primes, sim_values, yerr=sim_std_err,
                 label=r'alternative manager, approach 2, $\delta=%f$' % round(delta_t, 2))
    plt.plot(ror_primes, current_manager_values1, color='r', label=r'current manager, $P_\tau=120$')
    plt.plot(ror_primes, current_manager_values2, color='g', label=r'current manager, $P_\tau=180$')

    plt.legend()
    plt.show()

"""
This file performs the following tasks:

1. Initialize all necessary variables for the simulation.

2. Perform simulations.

3. Visualize results.

Copyright and Usage Information
===============================

This file is Copyright (c) 2023 Aarya Vatsa and Niall Whelan

Under the MIT License
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def calculate_expected_value(vol: float, t_h: int, ror: float, p_naught: float, p_tau: float, fee: float) -> float:
    """ Assuming that the largest value of NAV occurs at the final time step t_h, this function takes the NAV,
    that grows at rate 'ror' with volatility 'vol', and returns the expected value of our holdings at time t_h.

    :param vol: volatility
    :param t_h: time horizon
    :param ror: rate of return
    :param p_naught: value of the NAV at inception
    :param p_tau: value of the high water NAV (at time t_h)
    :param fee: the fee payment ratio
    :return: the expected value of our holdings
    """
    nav = p_naught * np.exp(ror * t_h)
    fee_payment = calculate_fees(vol, t_h, ror, p_naught, p_tau, fee)

    # In general, the expectation value of our holdings at time t = t_h consists of the NAV of the asset
    # less the fee payments.
    expected_value = nav - fee_payment
    return expected_value


def calculate_fees(vol: float, t_h: int, ror: float, p_naught: float, p_tau: float, fee: float) -> float:
    """ By assessing the performance at time t = t_h and assuming that the largest value of NAV occurs at t_h,
    this function returns the expected fee payment of the option.

    :param vol: volatility
    :param t_h: time horizon
    :param ror: rate of return
    :param p_naught: value of the NAV at inception
    :param p_tau: value of the high water NAV (at time t_h)
    :param fee: the fee payment ratio
    :return: the expected fee payment
    """
    N = norm.cdf  # define a normal distribution

    # Formula (5)
    d_pos_numerator = np.log(p_naught / p_tau) + t_h * (ror + vol ** 2 / 2)
    d_pos = d_pos_numerator / (vol * np.sqrt(t_h))

    d_neg_numerator = np.log(p_naught / p_tau) + t_h * (ror - vol ** 2 / 2)
    d_neg = d_neg_numerator / (vol * np.sqrt(t_h))

    fee_payment = fee * (N(d_pos) * p_naught * np.exp(ror * t_h) - N(d_neg) * p_tau)
    return max(fee_payment, 0.0)


def simulate_expected_value(num_simulations: int, vol: float, t_h: float, delta_t: float, ror: float, p_naught: float,
                            fee: float) -> float:
    """ By relaxing the assumption that the largest NAV will be observed at final time step t_h, this function uses
    a Monte Carlo method to calculate and return the expected value of our holdings.

    :param num_simulations: number of simulations to run
    :param vol: volatility
    :param t_h: time horizon
    :param delta_t: 'crystallization period', or time (in years) between observations of the NAV used
        for purposes of establishing the fee
    :param ror: rate of return
    :param p_naught: NAV at conception
    :param fee: the fee payment ratio
    :return: the simulated expected value of our holdings.
    """
    fees = []
    terminal_nav_values = []
    num_crystallization_periods = int(t_h / delta_t)

    for _ in range(num_simulations):  # number of paths
        curr_high_watermark = p_naught
        curr_p = p_naught
        for _ in range(num_crystallization_periods):  # timeseries for one path
            normal_rv = np.random.normal()
            exponential_term = delta_t * (ror - vol ** 2 / 2) + (normal_rv * vol * np.sqrt(delta_t))
            p_i = curr_p * np.exp(exponential_term)  # NAV at time i in timeseries
            if p_i > curr_high_watermark:
                curr_high_watermark = p_i
            curr_p = p_i

        path_fee = fee * max(curr_high_watermark - p_naught, 0.0)
        fees.append(path_fee)
        terminal_nav_values.append(curr_p)

    expected_values = [terminal_nav_values[i] - fees[i] for i in range(num_simulations)]
    avg_expected_value = np.average(expected_values)
    return avg_expected_value


if __name__ == '__main__':
    # declare variables
    volatility = 0.20
    time_horizon = 5
    rate_of_return = 0.05
    p_naught = 100
    fee = 0.20

    p_tau1 = 120
    p_tau2 = 180
    delta_t1 = 0.25
    delta_t2 = 1.0
    num_simulations = 10000  # number of simulations for Approach 2
    ror_primes = np.linspace(start=rate_of_return, stop=0.10, num=20)  # a range of r' for simulation

    # alternative manager expected value with both approaches
    alt_manager_values = [calculate_expected_value(volatility, time_horizon, ror, p_naught, p_naught, fee)
                          for ror in ror_primes]
    alt_manager_sim_1 = [simulate_expected_value(num_simulations, volatility, time_horizon, delta_t1,
                                                 ror, p_naught, fee) for ror in ror_primes]
    alt_manager_sim_2 = [simulate_expected_value(num_simulations, volatility, time_horizon, delta_t2,
                                                 ror, p_naught, fee) for ror in ror_primes]

    # current manager expected value with approach 1
    curr_manager_val_1 = calculate_expected_value(volatility, time_horizon, rate_of_return, p_naught, p_tau1, fee)
    curr_manager_val_2 = calculate_expected_value(volatility, time_horizon, rate_of_return, p_naught, p_tau2, fee)

    # labels
    plt.xlabel(r'$r^\prime$')
    plt.ylabel('Expected value')
    plt.title(r'Comparison for $T_H=%i$' % time_horizon + ', $\sigma=%1.1f$' % volatility)

    # plotting results
    plt.plot(ror_primes, alt_manager_values, label='alternative manager, approach 1')
    plt.plot(ror_primes, alt_manager_sim_1, label=r'alternative manager, approach 2, $\delta t=%1.2f$' % delta_t1)
    plt.plot(ror_primes, alt_manager_sim_2, label=r'alternative manager, approach 2, $\delta t=%1.2f$' % delta_t2)
    plt.axhline(curr_manager_val_1, color='r', linestyle='-', label=r'current manager, $P_{\tau}=%d$' % p_tau1)
    plt.axhline(curr_manager_val_2, color='r', linestyle='-', label=r'current manager, $P_{\tau}=%d$' % p_tau2)

    # visualizing graph
    plt.legend()
    plt.show()

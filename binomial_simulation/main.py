# run 5e6 iterations and save as .npy_
# random sample 10, 100, 1000, 10000, 100000, 1000000 returns, each with 5 times, avg(avg(sample_returns))
# plot as a line chart to show no convergence

import numpy as np
import DRAN_withKOKI_function

params = {
    "annual_coupon_rate": 3.65,
    "par_value": 1000,
    "total_days": 7,
    "index_starting_price": 0,
    "increase_rate": 0.4,
    "decrease_rate": -0.4,
    "upper_bound": 0.5,
    "lower_bound": -0.5,
    "movements_per_day": 24,
    "ko_boundary": 0.7,
    "ki_boundary": -0.7,
    "ki_coupon_rate": 3.65 / 2,
    "binomial_with_growth": False,
}

returns, in_bound_day_counts = DRAN_withKOKI_function.mc_DRAN_withKOKI(
    iterations=int(1e8), show_plot=False, **params
)

returns = np.array(returns)
np.save("./returns_1e8.npy", returns)

import numpy as np
import sympy as sp
import random


class Dran_KOKI_Increase:
    def __init__(
        self,
        annual_coupon_rate: float = 3.65,
        par_value: float = 1000,
        total_days: int = 7,
        index_starting_price: float = 0,
        daily_increase_mean: float = 0,
        daily_increase_sigma: float = 0.5,
        upper_bound: float = 0.5,
        lower_bound: float = -0.5,
        ko_boundary: float = 0.7,
        ki_boundary: float = -0.7,
        ki_annual_coupon_rate: float = 3.65 / 2,
        movements_per_day: int = 1,
    ):
        """
        Simulates daily random asset movements and calculates return and profit considering KO and KI boundaries.

        Parameters:
            annual_coupon_rate (float): Annual coupon rate as a fraction (e.g., 0.05 for 5%).
            par_value (float): Par value of the bond.
            total_days (int): Total number of days for the simulation.
            index_starting_price (float): Starting price of the index.
            daily_increase_mean (float): Daily increase mean: next day = today + daily increase.
            daily_increase_sigma (float): Daily increase sigma.
            upper_bound (float): Upper boundary for the index.
            lower_bound (float): Lower boundary for the index.
            ko_boundary (float): KO boundary for early stopping.
            ki_boundary (float): KI boundary for coupon adjustment.
            ki_coupon_rate (float): Coupon rate after KI boundary is triggered.
            movements_per_day (int): Number of movements per day.
            binomial_with_growth (bool): Whether to use binomial tree with growth.
        """
        self.annual_coupon_rate = annual_coupon_rate
        self.coupon_rate = annual_coupon_rate / 365
        self.ki_annual_coupon_rate = ki_annual_coupon_rate
        self.ki_coupon_rate = ki_annual_coupon_rate / 365
        self.par_value = par_value
        self.total_days = total_days
        self.total_steps = total_days * movements_per_day
        self.index_starting_price = index_starting_price
        self.daily_increase_mean = daily_increase_mean
        self.daily_increase_sigma = daily_increase_sigma
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.ki_boundary = ki_boundary
        self.ko_boundary = ko_boundary
        self.movements_per_day = movements_per_day

        self.increase_mean_per_step = daily_increase_mean / movements_per_day
        self.increase_sigma_per_step = daily_increase_sigma / movements_per_day

        self.index_price_array = np.array([self.index_starting_price])
        self.total_return = 0
        self.current_index_price = self.index_starting_price
        self.KI_TOUCHED = False
        self.KO_TOUCHED = False

    def reset(self):
        self.index_price_array = np.array([self.index_starting_price])
        self.total_return = 0
        self.current_index_price = self.index_starting_price
        self.KI_TOUCHED = False
        self.KO_TOUCHED = False

    def daily_step(self):
        actions = np.random.normal(
            self.increase_mean_per_step,
            self.increase_sigma_per_step,
            self.movements_per_day,
        )
        index_price_in_day = self.current_index_price + np.cumsum(actions)
        self.current_index_price = index_price_in_day[-1]

        if np.any(index_price_in_day >= self.ko_boundary):
            self.KO_TOUCHED = True
            return 0, index_price_in_day

        if np.any(index_price_in_day <= self.ki_boundary):
            self.KI_TOUCHED = True

        if np.all(index_price_in_day >= self.lower_bound) and np.all(
            index_price_in_day <= self.upper_bound
        ):
            if self.KI_TOUCHED:
                return self.ki_coupon_rate, index_price_in_day
            else:
                return self.coupon_rate, index_price_in_day
        return 0, index_price_in_day

    def run(self):
        daily_returns = []
        prices = [self.index_price_array]
        for i in range(self.total_days):
            daily_return, index_price_in_day = self.daily_step()
            daily_returns.append(daily_return)
            self.total_return += daily_return
            prices.append(index_price_in_day)
            if self.KO_TOUCHED:
                break

        self.index_price_array = np.concatenate(prices)
        return self.total_return, self.index_price_array


class Dran_KOKI_Multiply:
    def __init__(
        self,
        annual_coupon_rate: float = 3.65,
        par_value: float = 1000,
        total_days: int = 7,
        index_starting_price: float = 100,
        epsilon_sigma=0.01,
        epsilon_boundary: float = 0,
        daily_multiplier_sigma: float = 0.1,
        upper_bound: float = 120,
        lower_bound: float = 80,
        ko_boundary: float = 135,
        ki_boundary: float = 65,
        ki_annual_coupon_rate: float = 3.65 / 2,
        movements_per_day: int = 1,
    ):
        """
        Simulates daily random asset movements and calculates return and profit considering KO and KI boundaries.

        Parameters:
            annual_coupon_rate (float): Annual coupon rate as a fraction (e.g. 5 for 5%).
            par_value (float): Par value of the bond.
            total_days (int): Total number of days for the simulation.
            index_starting_price (float): Starting price of the index.
            daily_multiplier_mean (float): Daily multiplier mean: next day = today * daily multiplier.
            daily_multiplier_sigma (float): Daily multiplier sigma.
            upper_bound (float): Upper boundary for the index.
            lower_bound (float): Lower boundary for the index.
            ko_boundary (float): KO boundary for early stopping.
            ki_boundary (float): KI boundary for coupon adjustment.
            ki_coupon_rate (float): Coupon rate after KI boundary is triggered.
            movements_per_day (int): Number of movements per day.
            binomial_with_growth (bool): Whether to use binomial tree with growth.
        """
        self.annual_coupon_rate = annual_coupon_rate
        self.coupon_rate = annual_coupon_rate / 365
        self.ki_annual_coupon_rate = ki_annual_coupon_rate
        self.ki_coupon_rate = ki_annual_coupon_rate / 365
        self.par_value = par_value
        self.total_days = total_days
        self.total_steps = total_days * movements_per_day
        self.index_starting_price = index_starting_price
        self.daily_multiplier_sigma = daily_multiplier_sigma
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.ki_boundary = ki_boundary
        self.ko_boundary = ko_boundary
        self.movements_per_day = movements_per_day

        self.multiplier_mean_per_step = (
            np.exp(daily_multiplier_sigma / np.sqrt(movements_per_day)),
            np.exp(-daily_multiplier_sigma / np.sqrt(movements_per_day)),
        )
        self.epsilon_boundary = epsilon_boundary / movements_per_day
        self.epsilon_sigma = epsilon_sigma / movements_per_day

        self.index_price_array = np.array([self.index_starting_price])
        self.total_return = 0
        self.current_index_price = self.index_starting_price
        self.KI_TOUCHED = False
        self.KO_TOUCHED = False

    def reset(self):
        self.index_price_array = np.array([self.index_starting_price])
        self.total_return = 0
        self.current_index_price = self.index_starting_price
        self.KI_TOUCHED = False
        self.KO_TOUCHED = False

    def daily_step(self):
        actions = np.random.choice(
            self.multiplier_mean_per_step, self.movements_per_day
        )

        epsilons = np.random.normal(0, self.epsilon_sigma, self.movements_per_day).clip(
            -self.epsilon_boundary, self.epsilon_boundary
        )
        actions += epsilons
        index_price_in_day = self.current_index_price * np.cumprod(actions)
        self.current_index_price = index_price_in_day[-1]

        if np.any(index_price_in_day >= self.ko_boundary):
            self.KO_TOUCHED = True
            return 0, index_price_in_day

        if np.any(index_price_in_day <= self.ki_boundary):
            self.KI_TOUCHED = True

        if np.all(index_price_in_day >= self.lower_bound) and np.all(
            index_price_in_day <= self.upper_bound
        ):
            if self.KI_TOUCHED:
                return self.ki_coupon_rate, index_price_in_day
            else:
                return self.coupon_rate, index_price_in_day
        return 0, index_price_in_day

    def run(self):
        daily_returns = []
        prices = [self.index_price_array]
        for i in range(self.total_days):
            daily_return, index_price_in_day = self.daily_step()
            daily_returns.append(daily_return)
            self.total_return += daily_return
            prices.append(index_price_in_day)
            if self.KO_TOUCHED:
                break

        self.index_price_array = np.concatenate(prices)
        return self.total_return, self.index_price_array


def calculate_x_mean_variance_corrected(n, mean_y, var_y):
    """
    Calculate the mean (mu) and variance (sigma^2) of X given Y = X^n.

    Args:
    n (int): The power n in Y = X^n.
    mean_y (float): The mean of Y.
    var_y (float): The variance of Y.

    Returns:
    tuple: Mean (mu) and variance (sigma^2) of X.
    """
    # Define symbolic variables
    mu, sigma = sp.symbols("mu sigma", real=True, positive=True)

    # Approximate expressions for E[Y] and Var[Y] based on the normal moments
    if mean_y == 0:
        mean_y_expr = 0
        mu = 0
        var_y_expr = (n * sigma) ** 2  # Simplified assumption for Var[Y]
    else:
        mean_y_expr = mu**n  # Simplified assumption for E[Y]
        var_y_expr = (
            n * mu ** (n - 1) * sigma
        ) ** 2  # Simplified assumption for Var[Y]

    # Solve the equations for mu and sigma
    solutions = sp.solve(
        [sp.Eq(mean_y_expr, mean_y), sp.Eq(var_y_expr, var_y)], (mu, sigma)
    )

    mean = solutions[0][0]
    sigma = solutions[0][1]
    return mean, sigma

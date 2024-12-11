import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict


def DRAN(
    annual_coupon_rate: float,
    par_value: float,
    total_days: int,
    index_starting_price: float,
    sigma: float,
    upper_bound: float,
    lower_bound: float,
    movements_per_day: int = 1,
    binomial_with_growth: bool = True,
) -> Tuple[List[float], float, float, int]:
    """
    Simulates daily random asset movements and calculates return and profit.

    Parameters:
        annual_coupon_rate (float): Annual coupon rate as a fraction (e.g., 0.05 for 5%).
        par_value (float): Par value of the bond.
        total_days (int): Total number of days for the simulation.
        index_starting_price (float): Starting price of the index.
        sigma (float): Deviation per day
        upper_bound (float): Upper boundary for the index.
        lower_bound (float): Lower boundary for the index.
        movements_per_day (int): Number of movements per day.
        binomial_with_growth (bool): Whether to use binomial tree with growth.

    Returns:
        Tuple containing:
        - List of daily index values (List[float]).
        - Total return (float).
        - Total profit (float).
        - Number of days the index stayed within bounds (int).
    """
    # adjust total days to account for 0 indexing

    coupon_rate = annual_coupon_rate / 365
    total_steps = total_days * movements_per_day

    # Adjusted increase and decrease rates
    if binomial_with_growth:
        # is wrong, need to fix
        adjusted_increase_rate = np.exp(sigma * np.sqrt(movements_per_day))
        adjusted_decrease_rate = np.exp(-sigma * np.sqrt(movements_per_day))
    else:
        adjusted_increase_rate = sigma / np.sqrt(movements_per_day)
        adjusted_decrease_rate = -sigma / np.sqrt(movements_per_day)

    # Simulate index movements
    movements = np.random.choice(
        [adjusted_increase_rate, adjusted_decrease_rate], total_steps
    )
    index_list = [index_starting_price]

    for movement in movements:
        if binomial_with_growth:
            # Binomial tree with growth
            next_value = index_list[-1] * movement
        else:
            # Standard binomial tree
            next_value = index_list[-1] + movement

        index_list.append(np.round(next_value, 4))

    in_bound_day_count = 0
    # Check each day
    for day in range(total_days):
        start_idx = 1 + day * movements_per_day
        end_idx = start_idx + movements_per_day

        # Check if all values for the day are within the bounds
        daily_values = index_list[start_idx:end_idx]

        # If all values for the day are within the bounds
        if all(lower_bound <= value <= upper_bound for value in daily_values):
            in_bound_day_count += 1

    # Calculate return and profit
    ret = coupon_rate * in_bound_day_count
    profit = coupon_rate * par_value * in_bound_day_count

    return index_list, ret, profit, in_bound_day_count


def plot_DRAN(index_lists: List[List[float]], params: Dict):
    """
    Plots Monte Carlo simulations of index values with bounds.

    Parameters:
        index_lists (List[List[float]]): List of simulated index values.
        params (dict): Dictionary of parameters used for DRAN simulation.
    """
    lower_bound = params["lower_bound"]
    upper_bound = params["upper_bound"]
    movements_per_day = params["movements_per_day"]
    total_days = params["total_days"]

    plt.figure(figsize=(10, 6))

    # Generate x-values based on the total days and movements per day
    total_points = total_days * movements_per_day
    x_values = np.linspace(0, total_days, total_points + 1)

    plt.fill_between(x_values, lower_bound, upper_bound, color="lightgray", alpha=0.5)

    for index_list in index_lists:

        day_indices = np.linspace(0, total_days, len(index_list))

        # Plot the index values
        plt.plot(day_indices, index_list, color="black", alpha=0.6)

    # Plot bounds
    plt.axhline(y=upper_bound, color="red", linestyle="--", label="Upper Bound")
    plt.axhline(y=lower_bound, color="blue", linestyle="--", label="Lower Bound")

    # Set labels and title
    plt.xlabel("Day")
    plt.ylabel("Index Value")
    plt.title("Monte Carlo Simulation of Index Values with Bounds")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def mc_DRAN(
    iterations: int,
    show_plot: bool = True,
    **params,
) -> Tuple[List[float], List[int]]:
    """
    Performs Monte Carlo simulations and plots the results.

    Parameters:
        iterations (int): Number of Monte Carlo simulations to run.
        show_plot (bool): Whether to display the plot of simulations.
        params (dict): Parameters for the DRAN function.

    Returns:
        Tuple containing:
        - List of returns from simulations (List[float]).
        - List of in-bound day counts from simulations (List[int]).
    """
    all_index_lists = []
    returns = []
    in_bound_day_counts = []

    for _ in range(iterations):
        index_list, ret, _, in_bound_day_count = DRAN(**params)
        all_index_lists.append(index_list)
        returns.append(ret)
        in_bound_day_counts.append(in_bound_day_count)

    if show_plot:
        plot_DRAN(all_index_lists, params)

    return returns, in_bound_day_counts


def summarize_statistics(
    data: List[float], title: str, plot: bool = False, barplot: bool = False
):
    """
    Prints and optionally plots basic statistics for a dataset.

    Parameters:
        data (List[float]): Data to analyze.
        title (str): Title for the statistics summary and plots.
        plot (bool): Whether to display histograms and boxplots.
    """
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)

    print(f"\n{title} Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Standard Deviation: {std:.4f}")
    print(f"  Median: {median:.4f}")

    print(f"\n{title} Distribution:")
    print(pd.DataFrame(data).value_counts().sort_index())

    if plot:
        plt.figure(figsize=(12, 5))
        if barplot:
            from collections import Counter

            data_counts = Counter(data)
            plt.bar(
                data_counts.keys(),
                data_counts.values(),
                color="skyblue",
                edgecolor="black",
            )
            plt.title(f"{title} - Barplot")
        else:
            plt.hist(data, bins=20, alpha=0.7, color="blue", edgecolor="black")
            plt.title(f"{title} - Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


def run_mc_DRAN(
    iterations: int, show_plot: bool = True, summary: bool = False, **params
):
    returns, in_bound_day_counts = mc_DRAN(iterations, show_plot, **params)

    if summary:
        summarize_statistics(returns, title="DRAN Returns", plot=True, barplot=False)
        summarize_statistics(
            in_bound_day_counts,
            title="DRAN In Bound Day Counts (in days)",
            plot=True,
            barplot=True,
        )

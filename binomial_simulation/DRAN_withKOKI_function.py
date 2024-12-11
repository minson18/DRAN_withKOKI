import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict
from tqdm import tqdm
from multiprocessing import Pool


def DRAN_withKOKI(
    annual_coupon_rate: float,
    par_value: float,
    total_days: int,
    index_starting_price: float,
    sigma: float,
    upper_bound: float,
    lower_bound: float,
    ko_boundary: float,
    ki_boundary: float,
    ki_coupon_rate: float,
    movements_per_day: int = 1,
    binomial_with_growth: bool = True,
) -> Tuple[List[float], float, float, int]:
    """
    Simulates daily random asset movements and calculates return and profit considering KO and KI boundaries.

    Parameters:
        annual_coupon_rate (float): Annual coupon rate as a fraction (e.g., 0.05 for 5%).
        par_value (float): Par value of the bond.
        total_days (int): Total number of days for the simulation.
        index_starting_price (float): Starting price of the index.
        sigma (float): Deviation per day
        upper_bound (float): Upper boundary for the index.
        lower_bound (float): Lower boundary for the index.
        ko_boundary (float): KO boundary for early stopping.
        ki_boundary (float): KI boundary for coupon adjustment.
        ki_coupon_rate (float): Coupon rate after KI boundary is triggered.
        movements_per_day (int): Number of movements per day.
        binomial_with_growth (bool): Whether to use binomial tree with growth.

    Returns:
        Tuple containing:
        - List of daily index values (List[float]).
        - Total return (float).
        - Total profit (float).
        - Number of days the index stayed within bounds (int).
    """
    coupon_rate = annual_coupon_rate / 365
    ki_coupon_rate = ki_coupon_rate / 365
    total_steps = total_days * movements_per_day

    # Adjusted increase and decrease rates
    if binomial_with_growth:
        # is wrong, need to fix
        adjusted_increase_rate = np.exp(sigma / np.sqrt(movements_per_day))
        adjusted_decrease_rate = np.exp(-sigma / np.sqrt(movements_per_day))
    else:
        adjusted_increase_rate = sigma * np.sqrt(movements_per_day)
        adjusted_decrease_rate = -sigma * np.sqrt(movements_per_day)

    # Simulate index movements
    movements = np.random.choice(
        [adjusted_increase_rate, adjusted_decrease_rate], total_steps
    )
    index_list = [index_starting_price]

    in_bound_day_count_before_ki = 0
    in_bound_day_count_after_ki = 0
    ki_touched = False
    ko_touched = False

    for movement in movements:
        if binomial_with_growth:
            # Binomial tree with growth
            next_value = index_list[-1] * (1 + movement)
        else:
            # Standard binomial tree
            next_value = index_list[-1] + movement

        index_list.append(np.round(next_value, 4))

    # Check each day
    for day in range(total_days):
        start_idx = 1 + day * movements_per_day
        end_idx = start_idx + movements_per_day

        # Check if all values for the day are within the bounds
        daily_values = index_list[start_idx:end_idx]
        if any(value >= ko_boundary for value in daily_values) and not ko_touched:
            ko_touched = True
            break

        # Check if KI boundary is touched
        if any(value <= ki_boundary for value in daily_values) and not ki_touched:
            ki_touched = True

        # If all values for the day are within the bounds
        if all(lower_bound <= value <= upper_bound for value in daily_values):
            if ki_touched:
                in_bound_day_count_after_ki += 1
            else:
                in_bound_day_count_before_ki += 1

    # Calculate return and profit
    ret = (
        coupon_rate * in_bound_day_count_before_ki
        + ki_coupon_rate * in_bound_day_count_after_ki
    )
    profit = (
        coupon_rate * par_value * in_bound_day_count_before_ki
        + ki_coupon_rate * par_value * in_bound_day_count_after_ki
    )

    return (
        index_list,
        ret,
        profit,
        (in_bound_day_count_before_ki + in_bound_day_count_after_ki),
    )


def plot_DRAN_withKOKI(index_lists: List[List[float]], params: Dict):
    """
    Plots Monte Carlo simulations of index values with bounds, KO, and KI boundaries.

    Parameters:
        index_lists (List[List[float]]): List of simulated index values.
        params (dict): Dictionary of parameters used for DRAN simulation.
    """
    lower_bound = params["lower_bound"]
    upper_bound = params["upper_bound"]
    ko_boundary = params["ko_boundary"]
    ki_boundary = params["ki_boundary"]
    movements_per_day = params["movements_per_day"]
    total_days = params["total_days"]

    x_full = np.arange(0, total_days + 1 / movements_per_day, 1 / movements_per_day)

    plt.figure(figsize=(10, 6))

    for index_list in index_lists:
        # The index that the KO boundary is touched (+1 for indexing)
        ko_touched_idx = next(
            (i for i, value in enumerate(index_list) if value >= ko_boundary), None
        )

        if ko_touched_idx is not None:
            # Adjust the index list to cutoff at the KO boundary
            truncated_index_list = index_list[: ko_touched_idx + 1]
            x_truncated = x_full[: ko_touched_idx + 1]
        else:
            truncated_index_list = index_list
            x_truncated = x_full[: len(index_list)]
        # Plot the truncated index list with correct x-values
        plt.plot(x_truncated, truncated_index_list, color="black", alpha=0.6)

    # Plot bounds and KO, KI boundaries
    plt.axhline(y=upper_bound, color="red", linestyle="--", label="Upper Bound")
    plt.axhline(y=lower_bound, color="blue", linestyle="--", label="Lower Bound")
    plt.axhline(y=ko_boundary, color="green", linestyle="--", label="KO Boundary")
    plt.axhline(y=ki_boundary, color="orange", linestyle="--", label="KI Boundary")

    # Fill area between bounds
    plt.fill_between(
        x_full[: len(index_list)],
        lower_bound,
        upper_bound,
        color="lightgray",
        alpha=0.5,
    )

    plt.xlabel("Day")
    plt.ylabel("Index Value")
    plt.title(
        "Monte Carlo Simulation of Index Values with Bounds, KO, and KI Boundaries"
    )
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


# Move the run_simulation function outside the main function
def run_simulation(params):
    index_list, ret, _, in_bound_day_count = DRAN_withKOKI(**params)
    return index_list, ret, in_bound_day_count


def mc_DRAN_withKOKI(
    iterations: int,
    show_plot: bool = True,
    max_threads: int = None,
    **params,
) -> Tuple[List[float], List[int]]:
    """
    Performs Monte Carlo simulations and plots the results.

    Parameters:
        iterations (int): Number of Monte Carlo simulations to run.
        show_plot (bool): Whether to display the plot of simulations.
        max_threads (int): Maximum number of worker processes to use.
        params (dict): Parameters for the DRAN function.

    Returns:
        Tuple containing:
        - List of returns from simulations (List[float]).
        - List of in-bound day counts from simulations (List[int]).
    """
    all_index_lists = []
    returns = []
    in_bound_day_counts = []

    # Create a list of params to pass to the pool
    param_list = [params] * iterations

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(run_simulation, param_list), total=iterations))

    for index_list, ret, in_bound_day_count in results:
        all_index_lists.append(index_list)
        returns.append(ret)
        in_bound_day_counts.append(in_bound_day_count)

    if show_plot:
        plot_DRAN_withKOKI(all_index_lists, params)

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


def run_mc_DRAN_withKOKI(
    iterations: int, show_plot: bool = True, summary: bool = False, **params
):
    returns, in_bound_day_counts = mc_DRAN_withKOKI(iterations, show_plot, **params)

    if summary:
        summarize_statistics(returns, title="DRAN Returns", plot=True, barplot=False)
        summarize_statistics(
            in_bound_day_counts,
            title="DRAN In Bound Day Counts (in days and days^2)",
            plot=True,
            barplot=True,
        )

# ============================================================
# MAT2007 Introduction to Programming - Final Project
#
# Research question:
# Which major metals have experienced the greatest real-price
# volatility since 1900, and has their volatility changed
# over time?
#
# Dataset:
# Our World in Data - Real commodity price index, metals
# Original source: Jacks (2019)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

FILENAME = "real-commodity-price-index-metals.csv"

METALS = [
    "Iron ore",
    "Bauxite",
    "Tin",
    "Zinc",
    "Steel",
    "Manganese",
    "Aluminum",
    "Chromium",
    "Copper",
    "Lead",
    "Nickel"
]

# Common period for all metals.
START_YEAR = 1900
END_YEAR = 2020

# Two approximately equal periods for comparison.
FIRST_PERIOD_END = 1959
SECOND_PERIOD_START = 1960


# ------------------------------------------------------------
# 2. LOAD AND CHECK DATA
# ------------------------------------------------------------

def load_data(filename):
    """
    Read the CSV file and perform basic checks.
    """

    # Check that the file exists before trying to open it.
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Error: '{filename}' was not found."
        )

    try:
        data = pd.read_csv(filename)

    except pd.errors.EmptyDataError:
        raise ValueError(
            "Error: the CSV file is empty."
        )

    except pd.errors.ParserError:
        raise ValueError(
            "Error: the CSV file could not be read correctly."
        )

    # Check that all columns needed for the analysis exist.
    required_columns = ["Year"] + METALS

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Error: missing columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError(
            "Error: the dataset contains no observations."
        )

    return data


# ------------------------------------------------------------
# 3. PREPARE DATA
# ------------------------------------------------------------

def prepare_data(data):
    """
    Select the common 1900-2020 period and keep only
    the columns needed for the analysis.
    """

    common_data = data[
        (data["Year"] >= START_YEAR) &
        (data["Year"] <= END_YEAR)
    ].copy()

    common_data = common_data[
        ["Year"] + METALS
    ]

    if common_data.empty:
        raise ValueError(
            f"No observations found between "
            f"{START_YEAR} and {END_YEAR}."
        )

    # Check for duplicate years
    if common_data["Year"].duplicated().any():
        raise ValueError("Duplicate years detected.")

    # Make sure observations are in chronological order.
    common_data = common_data.sort_values("Year")
    common_data = common_data.reset_index(drop=True)

    # Check that all years are present
    expected_years = np.arange(START_YEAR, END_YEAR + 1)

    if not np.array_equal(
        common_data["Year"].to_numpy(),
        expected_years
    ):
        raise ValueError("Missing or non-consecutive years.")

    # Check that price indices are valid
    if common_data[METALS].isna().any().any():
        raise ValueError("Missing metal price values detected.")

    if (common_data[METALS] <= 0).any().any():
        raise ValueError("Price indices must be positive.")


    return common_data


# ------------------------------------------------------------
# 4. CALCULATE ANNUAL PERCENTAGE CHANGES
# ------------------------------------------------------------

def calculate_annual_changes(data):
    """
    Calculate year-to-year percentage changes.

    Percentage change =
    (current value - previous value) / previous value * 100
    """

    annual_changes = pd.DataFrame()

    annual_changes["Year"] = data["Year"]

    for metal in METALS:

        annual_changes[metal] = (
            data[metal].pct_change(fill_method=None) * 100
        )

    return annual_changes


# ------------------------------------------------------------
# 5. CALCULATE UNCERTAINTY OF VOLATILITY
# ------------------------------------------------------------

def volatility_uncertainty(volatility, n):
    """
    Estimate the standard error of a sample standard deviation.

    Approximation:

        SE(s) = s / sqrt(2 * (N - 1))

    where:
        s = sample standard deviation
        N = number of observations

    This approximation assumes independent observations
    from an approximately normal distribution.
    """

    # At least two observations are needed to calculate
    # a sample standard deviation.
    if n < 2:
        return np.nan

    uncertainty = volatility / np.sqrt(
        2 * (n - 1)
    )

    return uncertainty


# ------------------------------------------------------------
# 6. CALCULATE OVERALL STATISTICS
# ------------------------------------------------------------

def calculate_statistics(annual_changes):
    """
    Calculate statistics for each metal.

    Standard deviation of annual percentage changes is used
    as the measure of price volatility.
    """

    results = []

    for metal in METALS:

        # Remove missing observations for this metal.
        changes = annual_changes[
            ["Year", metal]
        ].dropna()

        if changes.empty:
            continue

        n = len(changes)

        # Mean annual percentage change.
        mean_change = changes[metal].mean()

        # Standard deviation = volatility.
        volatility = changes[metal].std()

        # Statistical uncertainty of volatility.
        uncertainty = volatility_uncertainty(
            volatility,
            n
        )

        # Shape of the annual price-change distribution
        skewness = changes[metal].skew()
        excess_kurtosis = changes[metal].kurt()


        # Largest annual increase and decrease.
        maximum_change = changes[metal].max()
        minimum_change = changes[metal].min()

        # Find years in which these occurred.
        max_year = int(
            changes.loc[
                changes[metal].idxmax(),
                "Year"
            ]
        )

        min_year = int(
            changes.loc[
                changes[metal].idxmin(),
                "Year"
            ]
        )

        results.append({
            "Metal": metal,
            "N": n,
            "Mean annual change (%)": mean_change,
            "Volatility SD (%)": volatility,
            "Volatility uncertainty (%)": uncertainty,
            "Maximum increase (%)": maximum_change,
            "Year of maximum increase": max_year,
            "Maximum decrease (%)": minimum_change,
            "Year of maximum decrease": min_year,
            "Skewness": skewness,
            "Excess kurtosis": excess_kurtosis,
        })

    statistics = pd.DataFrame(results)

    # Rank from most volatile to least volatile.
    statistics = statistics.sort_values(
        "Volatility SD (%)",
        ascending=False
    )

    statistics = statistics.reset_index(drop=True)

    return statistics


# ------------------------------------------------------------
# 7. COMPARE TWO TIME PERIODS
# ------------------------------------------------------------

def compare_periods(annual_changes):
    """
    Compare volatility between 1901-1959 and 1960-2020.

    Uncertainty is also calculated for each volatility
    estimate.
    """

    results = []

    first_period = annual_changes[
        (annual_changes["Year"] >= START_YEAR) &
        (annual_changes["Year"] <= FIRST_PERIOD_END)
    ]

    second_period = annual_changes[
        (annual_changes["Year"] >= SECOND_PERIOD_START) &
        (annual_changes["Year"] <= END_YEAR)
    ]

    for metal in METALS:

        first_values = first_period[metal].dropna()
        second_values = second_period[metal].dropna()

        first_n = len(first_values)
        second_n = len(second_values)

        first_volatility = first_values.std()
        second_volatility = second_values.std()

        first_uncertainty = volatility_uncertainty(
            first_volatility,
            first_n
        )

        second_uncertainty = volatility_uncertainty(
            second_volatility,
            second_n
        )

        # Percentage change in volatility between periods.
        if first_volatility != 0:

            volatility_change = (
                (second_volatility - first_volatility)
                / first_volatility
            ) * 100

        else:
            volatility_change = np.nan

        results.append({
            "Metal": metal,

            "1901-1959 volatility (%)":
                first_volatility,

            "1901-1959 uncertainty (%)":
                first_uncertainty,

            "1960-2020 volatility (%)":
                second_volatility,

            "1960-2020 uncertainty (%)":
                second_uncertainty,

            "Change in volatility (%)":
                volatility_change
        })

    comparison = pd.DataFrame(results)

    return comparison


# ------------------------------------------------------------
# 8. DECADAL SUB-SAMPLING
# ------------------------------------------------------------

def calculate_decadal_volatility(annual_changes):
    """
    Divide annual percentage changes into non-overlapping
    10-year subsamples.

    The subsamples are:
        1901-1910
        1911-1920
        ...
        2011-2020

    Volatility and its uncertainty are calculated separately
    for each decade and each metal.
    """

    results = []

    # Annual changes begin in 1901 because the 1900
    # observation has no previous year.
    for decade_start in range(1901, 2020, 10):

        decade_end = decade_start + 9

        # Select one 10-year subsample.
        subsample = annual_changes[
            (annual_changes["Year"] >= decade_start) &
            (annual_changes["Year"] <= decade_end)
        ]

        for metal in METALS:

            values = subsample[metal].dropna()

            n = len(values)

            # Need at least two observations for SD.
            if n < 2:
                continue

            volatility = values.std()

            uncertainty = volatility_uncertainty(
                volatility,
                n
            )

            results.append({
                "Period":
                    f"{decade_start}-{decade_end}",

                "Start year":
                    decade_start,

                "Metal":
                    metal,

                "N":
                    n,

                "Volatility (%)":
                    volatility,

                "Uncertainty (%)":
                    uncertainty
            })

    decadal_results = pd.DataFrame(results)

    return decadal_results


# ------------------------------------------------------------
# 9. PLOT PRICE HISTORY
# ------------------------------------------------------------

def plot_price_history(data):
    """
    Plot real metal price indices from 1900-2020.
    """

    plt.figure(figsize=(12, 7))

    for metal in METALS:

        plt.plot(
            data["Year"],
            data[metal],
            label=metal
        )

    plt.title(
        "Real Metal Commodity Price Indices, 1900-2020"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Real price index (1900 = 100)"
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "metal_price_history.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 10. PLOT OVERALL VOLATILITY WITH UNCERTAINTIES
# ------------------------------------------------------------

def plot_overall_volatility(statistics):
    """
    Plot overall volatility for each metal.

    Error bars show the approximate statistical uncertainty
    of each standard-deviation estimate.
    """

    plot_data = statistics.sort_values(
        "Volatility SD (%)",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["Metal"],
        plot_data["Volatility SD (%)"],
        xerr=plot_data["Volatility uncertainty (%)"],
        capsize=4
    )

    plt.title(
        "Volatility of Annual Real Metal Price Changes, "
        "1901-2020"
    )

    plt.xlabel(
        "Standard deviation of annual price changes (%)"
    )

    plt.ylabel("Metal")

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "metal_volatility_with_uncertainty.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 11. PLOT TWO-PERIOD COMPARISON WITH UNCERTAINTIES
# ------------------------------------------------------------

def plot_period_comparison(comparison):
    """
    Compare volatility in 1901-1959 and 1960-2020.

    Error bars show approximate uncertainty.
    """

    metals = comparison["Metal"]

    # Numerical positions for all metals on the x-axis.
    x = np.arange(len(metals))

    width = 0.35

    plt.figure(figsize=(12, 7))

    # First period
    plt.bar(
        x - width / 2,
        comparison["1901-1959 volatility (%)"],
        width,
        yerr=comparison["1901-1959 uncertainty (%)"],
        capsize=3,
        label="1901-1959"
    )

    # Second period
    plt.bar(
        x + width / 2,
        comparison["1960-2020 volatility (%)"],
        width,
        yerr=comparison["1960-2020 uncertainty (%)"],
        capsize=3,
        label="1960-2020"
    )

    plt.title(
        "Metal Price Volatility: "
        "1901-1959 vs 1960-2020"
    )

    plt.xlabel("Metal")

    plt.ylabel(
        "Standard deviation of annual price changes (%)"
    )

    plt.xticks(
        x,
        metals,
        rotation=45,
        ha="right"
    )

    plt.legend(title="Period")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "volatility_period_comparison_with_uncertainty.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 12. PLOT DECADAL SUB-SAMPLES
# ------------------------------------------------------------

def plot_decadal_volatility(decadal_results):
    """
    Plot volatility across non-overlapping 10-year subsamples.

    Four metals are shown to keep the graph readable:
    Zinc      - highest overall estimated volatility
    Nickel    - largest increase between the two broad periods
    Iron ore  - large decrease between the two broad periods
    Bauxite   - lowest overall estimated volatility
    """

    selected_metals = [
        "Zinc",
        "Nickel",
        "Iron ore",
        "Bauxite"
    ]

    plt.figure(figsize=(12, 7))

    for metal in selected_metals:

        metal_data = decadal_results[
            decadal_results["Metal"] == metal
        ]

        plt.errorbar(
            metal_data["Period"],
            metal_data["Volatility (%)"],
            yerr=metal_data["Uncertainty (%)"],
            marker="o",
            capsize=3,
            label=metal
        )

    plt.title(
        "Metal Price Volatility Across 10-Year Subsamples"
    )

    plt.xlabel("10-year subsample")

    plt.ylabel(
        "Standard deviation of annual price changes (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "decadal_volatility.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 13. HISTOGRAM OF MOST VOLATILE METAL
# ------------------------------------------------------------

def plot_most_volatile_histogram(
        annual_changes,
        statistics):
    """
    Plot the distribution of annual percentage changes
    for the most volatile metal.
    """

    most_volatile = statistics.iloc[0]["Metal"]

    changes = annual_changes[
        most_volatile
    ].dropna()

    plt.figure(figsize=(9, 6))

    plt.hist(
        changes,
        bins=20,
        edgecolor="black"
    )

    plt.title(
        f"Distribution of Annual Price Changes: "
        f"{most_volatile}"
    )

    plt.xlabel(
        "Annual price change (%)"
    )

    plt.ylabel("Frequency")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "most_volatile_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 14. PRINT RESULTS
# ------------------------------------------------------------

def print_results(
        statistics,
        comparison,
        decadal_results):
    """
    Print the main numerical results.
    """

    print("\n" + "=" * 75)
    print("OVERALL VOLATILITY RESULTS: 1901-2020")
    print("=" * 75)

    print(
        statistics.round(2).to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("VOLATILITY COMPARISON BETWEEN PERIODS")
    print("=" * 75)

    print(
        comparison.round(2).to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("DECADAL SUB-SAMPLE RESULTS")
    print("=" * 75)

    print(
        decadal_results.round(2).to_string(
            index=False
        )
    )

    # Highest and lowest overall volatility.
    most_volatile = statistics.iloc[0]

    least_volatile = statistics.iloc[-1]

    print("\n" + "=" * 75)
    print("MAIN FINDINGS")
    print("=" * 75)

    print(
        f"Most volatile metal: "
        f"{most_volatile['Metal']} "
        f"= {most_volatile['Volatility SD (%)']:.2f} "
        f"+/- "
        f"{most_volatile['Volatility uncertainty (%)']:.2f}%"
    )

    print(
        f"Least volatile metal: "
        f"{least_volatile['Metal']} "
        f"= {least_volatile['Volatility SD (%)']:.2f} "
        f"+/- "
        f"{least_volatile['Volatility uncertainty (%)']:.2f}%"
    )

    # Largest increase in volatility.
    largest_increase = comparison.loc[
        comparison[
            "Change in volatility (%)"
        ].idxmax()
    ]

    # Largest decrease in volatility.
    largest_decrease = comparison.loc[
        comparison[
            "Change in volatility (%)"
        ].idxmin()
    ]

    print(
        f"Largest increase in volatility: "
        f"{largest_increase['Metal']} "
        f"({largest_increase['Change in volatility (%)']:.2f}%)"
    )

    print(
        f"Largest decrease in volatility: "
        f"{largest_decrease['Metal']} "
        f"({largest_decrease['Change in volatility (%)']:.2f}%)"
    )


# ------------------------------------------------------------
# 15. SAVE RESULTS
# ------------------------------------------------------------

def save_results(
        statistics,
        comparison,
        decadal_results):
    """
    Save all calculated results as CSV files.
    """

    statistics.to_csv(
        "overall_volatility_results.csv",
        index=False
    )

    comparison.to_csv(
        "period_volatility_results.csv",
        index=False
    )

    decadal_results.to_csv(
        "decadal_volatility_results.csv",
        index=False
    )


# ------------------------------------------------------------
# 16. MAIN PROGRAM
# ------------------------------------------------------------

def main():
    """
    Run the complete analysis.
    """

    print("Loading data...")

    # Load original dataset.
    data = load_data(FILENAME)

    print(
        f"Dataset loaded successfully: "
        f"{len(data)} observations."
    )

    # Select common analysis period.
    common_data = prepare_data(data)

    print(
        f"Analysis period: "
        f"{common_data['Year'].min()}-"
        f"{common_data['Year'].max()}"
    )

    # Calculate annual percentage changes.
    annual_changes = calculate_annual_changes(
        common_data
    )

    # Calculate overall volatility.
    statistics = calculate_statistics(
        annual_changes
    )

    # Compare first and second periods.
    comparison = compare_periods(
        annual_changes
    )

    # Perform decadal sub-sampling.
    decadal_results = calculate_decadal_volatility(
        annual_changes
    )

    # Print numerical results.
    print_results(
        statistics,
        comparison,
        decadal_results
    )

    # Save numerical results.
    save_results(
        statistics,
        comparison,
        decadal_results
    )

    # Produce all figures.
    plot_price_history(
        common_data
    )

    plot_overall_volatility(
        statistics
    )

    plot_period_comparison(
        comparison
    )

    plot_decadal_volatility(
        decadal_results
    )

    plot_most_volatile_histogram(
        annual_changes,
        statistics
    )

    print("\nAnalysis complete.")


# Run main() only when this Python file is executed directly.
if __name__ == "__main__":
    main()
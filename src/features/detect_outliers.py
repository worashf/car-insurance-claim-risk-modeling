import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
import pandas as pd


def detect_outliers(
        df: pd.DataFrame,
        numerical_cols: List[str],
        figsize: tuple = (12, 6),
        whis: float = 1.5,
        show_fliers: bool = True,
        show_stats: bool = True,
        palette: str = "Blues"
) -> None:
    """
    Detect and visualize outliers in numerical columns using box plots and statistical analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the data
    numerical_cols : List[str]
        List of numerical columns to analyze for outliers
    figsize : tuple, optional
        Size of the figures (default: (12, 6))
    whis : float, optional
        Parameter for boxplot whisker length (default: 1.5)
    show_fliers : bool, optional
        Whether to show outliers in the plot (default: True)
    show_stats : bool, optional
        Whether to print outlier statistics (default: True)
    palette : str, optional
        Color palette for plots (default: "Blues")

    Returns:
    --------
    None
    """

    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    for col in numerical_cols:
        # Calculate outlier thresholds
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - whis * iqr
        upper_bound = q3 + whis * iqr

        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df)) * 100

        if show_stats:
            print(f"\n📊 Outlier Analysis for {col}:")
            print(f"• Lower bound: {lower_bound:.2f}")
            print(f"• Upper bound: {upper_bound:.2f}")
            print(f"• Number of outliers: {len(outliers)} ({outlier_percentage:.2f}% of data)")
            if not outliers.empty:
                print(f"• Min outlier value: {outliers[col].min():.2f}")
                print(f"• Max outlier value: {outliers[col].max():.2f}")

        # Create visualization
        plt.figure(figsize=figsize)

        # Boxplot
        ax = sns.boxplot(
            x=df[col],
            whis=whis,
            showfliers=show_fliers,
            color=sns.color_palette(palette)[2],
            width=0.4
        )

        # Add annotations
        plt.title(f'Outlier Detection for {col}\n({len(outliers)} outliers, {outlier_percentage:.1f}%)', pad=20)
        plt.xlabel(col)

        # Add reference lines
        plt.axvline(lower_bound, color='red', linestyle='--', alpha=0.5, label='Lower Bound')
        plt.axvline(upper_bound, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
        plt.axvline(df[col].median(), color='green', linestyle='-', alpha=0.5, label='Median')

        # Add legend
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Violin plot for additional distribution context
        plt.figure(figsize=figsize)
        sns.violinplot(
            x=df[col],
            color=sns.color_palette(palette)[1],
            inner="quartile"
        )
        plt.title(f'Distribution with Outliers for {col}', pad=20)
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()


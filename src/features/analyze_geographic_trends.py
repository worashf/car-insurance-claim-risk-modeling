import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union
import pandas as pd
import numpy as np


def analyze_geographic_trends(
        df: pd.DataFrame,
        geographic_col: str = 'Province',
        metric_cols: List[str] = ['TotalPremium', 'TotalClaims'],
        aggregation: str = 'mean',
        top_n: Optional[int] = None,
        figsize: tuple = (14, 8),
        rotation: int = 45,
        palette: str = "viridis",
        show_stats: bool = True,
        sort_values: bool = True
) -> None:
    """
    Analyze and visualize trends across geographic regions with multiple metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the data
    geographic_col : str, optional
        Column containing geographic divisions (default: 'Province')
    metric_cols : List[str], optional
        List of metric columns to analyze (default: ['TotalPremium', 'TotalClaims'])
    aggregation : str, optional
        Aggregation function ('mean', 'median', 'sum', 'count') (default: 'mean')
    top_n : Optional[int], optional
        Show only top N regions by metric (default: None shows all)
    figsize : tuple, optional
        Size of the figures (default: (14, 8))
    rotation : int, optional
        Rotation angle for x-axis labels (default: 45)
    palette : str, optional
        Color palette for plots (default: "viridis")
    show_stats : bool, optional
        Whether to print statistical summary (default: True)
    sort_values : bool, optional
        Whether to sort geographic regions by metric (default: True)

    Returns:
    --------
    None
    """

    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Validate aggregation function
    agg_funcs = {
        'mean': np.mean,
        'median': np.median,
        'sum': np.sum,
        'count': 'count'
    }
    if aggregation not in agg_funcs:
        raise ValueError(f"Aggregation must be one of {list(agg_funcs.keys())}")

    for metric in metric_cols:
        # Calculate aggregated values
        agg_df = df.groupby(geographic_col)[metric].agg(agg_funcs[aggregation])

        if sort_values:
            agg_df = agg_df.sort_values(ascending=False)

        if top_n:
            agg_df = agg_df.head(top_n)

        if show_stats:
            print(f"\n📊 Geographic Trends for {metric} ({aggregation}) by {geographic_col}:")
            print(agg_df.describe().to_string())
            print("\nTop Regions:")
            print(agg_df.head(5).to_string())
            print("\nBottom Regions:")
            print(agg_df.tail(5).to_string())

        # Create visualization
        plt.figure(figsize=figsize)

        # Bar plot
        ax = sns.barplot(
            x=agg_df.index,
            y=agg_df.values,
            palette=palette,
            order=agg_df.index if sort_values else None
        )

        plt.title(f'{metric} {aggregation.title()} by {geographic_col}', pad=20)
        plt.ylabel(f'{aggregation.title()} {metric}')
        plt.xlabel(geographic_col)

        # Rotate x-axis labels if needed
        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')

        # Add value annotations
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.1f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points'
            )

        plt.tight_layout()
        plt.show()

        # Boxplot for distribution across regions
        plt.figure(figsize=figsize)

        # Get top regions for focused analysis
        plot_df = df[df[geographic_col].isin(agg_df.index)]

        sns.boxplot(
            data=plot_df,
            x=geographic_col,
            y=metric,
            palette=palette,
            order=agg_df.index if sort_values else None
        )

        plt.title(f'Distribution of {metric} across {geographic_col}', pad=20)
        plt.ylabel(metric)
        plt.xlabel(geographic_col)

        if rotation:
            plt.xticks(rotation=rotation, ha='right')

        plt.tight_layout()
        plt.show()

# Example usage:
# analyze_geographic_trends(
#     df,
#     geographic_col='Province',
#     metric_cols=['TotalPremium', 'TotalClaims', 'LossRatio'],
#     aggregation='median',
#     top_n=10,
#     palette="mako",
#     rotation=60
# )
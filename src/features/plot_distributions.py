import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import pandas as pd


def plot_distributions(
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        figsize: tuple = (10, 6),
        rotation: int = 45,
        palette: str = "viridis"
) -> None:
    """
    Plot distributions for numerical and categorical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the data
    numerical_cols : List[str], optional
        List of numerical columns to plot (default: None)
    categorical_cols : List[str], optional
        List of categorical columns to plot (default: None)
    figsize : tuple, optional
        Size of the figures (default: (10, 6))
    rotation : int, optional
        Rotation angle for x-axis labels (default: 45)
    palette : str, optional
        Color palette for plots (default: "viridis")

    Returns:
    --------
    None
    """

    # Set style
    sns.set(style="whitegrid")

    # Plot numerical distributions
    if numerical_cols:
        for col in numerical_cols:
            plt.figure(figsize=figsize)
            sns.histplot(
                data=df,
                x=col,
                bins=30,
                kde=True,
                color=plt.get_cmap(palette)(0.5)
            )
            plt.title(f'Distribution of {col}', pad=20)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

            # Boxplot for numerical column
            plt.figure(figsize=figsize)
            sns.boxplot(
                data=df,
                x=col,
                color=plt.get_cmap(palette)(0.3)
            )
            plt.title(f'Boxplot of {col}', pad=20)
            plt.tight_layout()
            plt.show()

    # Plot categorical distributions
    if categorical_cols:
        for col in categorical_cols:
            # Skip if too many categories
            if df[col].nunique() > 20:
                print(f"Skipping {col} - too many categories ({df[col].nunique()})")
                continue

            plt.figure(figsize=figsize)
            ax = sns.countplot(
                data=df,
                x=col,
                palette=palette,
                order=df[col].value_counts().index
            )
            plt.title(f'Distribution of {col}', pad=20)
            plt.xlabel(col)
            plt.ylabel('Count')

            # Rotate x-axis labels if needed
            if rotation:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')

            # Add percentage annotations
            total = len(df)
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_height() + 0.01 * total
                ax.annotate(percentage, (x, y), ha='center')

            plt.tight_layout()
            plt.show()

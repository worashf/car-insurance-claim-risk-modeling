import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec


def create_insightful_visuals(df: pd.DataFrame) -> None:
    """
    Create three professional visualizations highlighting key insurance insights:
    1. Loss Ratio by Province (with profitability markers)
    2. Temporal trends in claims (dual-axis for frequency and severity)
    3. Vehicle make risk profile (normalized by frequency)
    """

    # Set professional style
    sns.set_theme(style="whitegrid")
    sns.set_palette("viridis")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)

    # ----------------------------------
    # Plot 1: Loss Ratio by Province
    # ----------------------------------
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate loss ratio safely (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['LossRatio'] = np.where(
            df['TotalPremium'] > 0,
            df['TotalClaims'] / df['TotalPremium'],
            np.nan  # Set to NaN if premium is zero
        )

    # Filter out NaN values before grouping
    province_loss = df[df['LossRatio'].notna()].groupby('Province')['LossRatio'].mean().sort_values()

    # Create bar plot with reference lines
    # It's good practice to ensure consistent coloring if not mapping to a hue.
    # Using a single color from the palette for simplicity here.
    bars = ax1.barh(province_loss.index, province_loss.values * 100,
                    color=sns.color_palette("viridis", 1)[0])

    # Add reference lines and annotations
    ax1.axvline(100, color='red', linestyle='--', alpha=0.7, label='Break-even (100%)')
    ax1.axvline(75, color='orange', linestyle=':', alpha=0.7, label='Profitability Threshold (75%)')

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%',
                 va='center', ha='left')

    ax1.set_title('Insurance Loss Ratio by Province\n(Claims/Premium)', pad=20)
    ax1.set_xlabel('Loss Ratio (%)')
    ax1.xaxis.set_major_formatter(PercentFormatter())
    ax1.legend()

    # ----------------------------------
    # Plot 2: Temporal Trends (Dual Axis)
    # ----------------------------------
    ax2 = fig.add_subplot(gs[1, :])

    # Prepare temporal data
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    # Group by period and calculate mean for TotalClaims > 0
    monthly_data = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg(
        ClaimCount=('TotalClaims', 'count'),
        AvgClaim=('TotalClaims', lambda x: x[x > 0].mean()) # Ensure average is for claims > 0
    ).reset_index()
    monthly_data['Month'] = monthly_data['TransactionMonth'].astype(str)

    # Create twin axes
    ax2b = ax2.twinx()

    # Plot claim frequency
    sns.lineplot(data=monthly_data, x='Month', y='ClaimCount',
                 ax=ax2, color='teal', marker='o', label='Claim Frequency', sort=False) # sort=False for period type

    # Plot average claim amount
    sns.lineplot(data=monthly_data, x='Month', y='AvgClaim',
                 ax=ax2b, color='coral', marker='s', label='Average Claim Amount', sort=False) # sort=False

    ax2.set_title('Temporal Trends in Insurance Claims', pad=20)
    ax2.set_ylabel('Number of Claims', color='teal')
    ax2b.set_ylabel('Average Claim Amount (R)', color='coral')

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Rotate x-axis labels
    ax2.tick_params(axis='x', rotation=45)

    # ----------------------------------
    # Plot 3: Vehicle Make Risk Profile
    # ----------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # Prepare vehicle data - filter out zero claims
    vehicle_data = df[df['TotalClaims'] > 0].copy()
    top_makes = vehicle_data['make'].value_counts().nlargest(10).index
    vehicle_data = vehicle_data[vehicle_data['make'].isin(top_makes)]

    # Plot 3a: Claim frequency by make
    make_counts = vehicle_data['make'].value_counts().sort_values()
    make_counts.plot(kind='barh', ax=ax3, color=sns.color_palette("rocket", len(make_counts)))
    ax3.set_title('Top 10 Vehicle Makes by Claim Frequency')
    ax3.set_xlabel('Number of Claims')

    # Plot 3b: Normalized claim amount (filter extreme values)
    sns.boxplot(data=vehicle_data, y='make', x='TotalClaims',
                ax=ax4, order=make_counts.index,
                showfliers=False, palette="rocket_r",
                hue='make', legend=False)

    ax4.set_title('Claim Amount Distribution by Make')
    ax4.set_xlabel('Claim Amount (R)')
    ax4.set_ylabel('')

    # Add super title
    fig.suptitle('Key Insurance Portfolio Insights', y=1.02, fontsize=16, fontweight='bold')



    plt.show()
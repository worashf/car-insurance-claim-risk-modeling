import  pandas as pd
import numpy as np

def calculate_aggregated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Claim Frequency, Claim Severity, and Margin grouped by Province, PostalCode, and Gender.

    Args:
        df (pd.DataFrame): Input DataFrame with 'TotalPremium' and 'TotalClaims'.

    Returns:
        pd.DataFrame: Aggregated metrics.
    """
    df = df.copy()
    df['HasClaim'] = df['TotalClaims'] > 0

    aggregated_df = df.groupby(['Province', 'PostalCode', 'Gender']).agg(
        ClaimFrequency=('HasClaim', 'mean'),
        TotalPolicies=('HasClaim', 'size'),
        ClaimCount=('HasClaim', 'sum'),
        TotalPremiums=('TotalPremium', 'sum'),
        TotalClaimAmounts=('TotalClaims', 'sum')
    ).reset_index()

    aggregated_df['ClaimSeverity'] = np.where(
        aggregated_df['ClaimCount'] > 0,
        aggregated_df['TotalClaimAmounts'] / aggregated_df['ClaimCount'],
        0.0
    )
    # Just in case, still fill any NaN
    aggregated_df['ClaimSeverity'] = aggregated_df['ClaimSeverity'].fillna(0.0)
    aggregated_df['Margin'] = aggregated_df['TotalPremiums'] - aggregated_df['TotalClaimAmounts']

    return aggregated_df


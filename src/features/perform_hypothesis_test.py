import pandas as pd
import numpy as np
from scipy import stats

## Statistical Testing Functions

def perform_hypothesis_test(group1_df: pd.DataFrame, group2_df: pd.DataFrame, metric_name: str, test_type: str):
    """
    Performs a statistical hypothesis test between two groups for a given metric.

    Args:
        group1_df (pd.DataFrame): DataFrame for Group 1.
        group2_df (pd.DataFrame): DataFrame for Group 2.
        metric_name (str): The name of the metric to test ('ClaimFrequency', 'ClaimSeverity', 'Margin').
        test_type (str): 't_test' for numerical, 'chi2_test' for categorical (frequency).

    Returns:
        float: The p-value from the statistical test, or np.nan if conditions for the test are not met.
    """
    if test_type == 't_test':
        # Ensure there's enough data in both groups for a t-test
        if len(group1_df[metric_name]) < 2 or len(group2_df[metric_name]) < 2:
            return np.nan # Not enough data for comparison
        t_statistic, p_value = stats.ttest_ind(group1_df[metric_name], group2_df[metric_name], equal_var=False) # Welch's t-test
        return p_value
    elif test_type == 'chi2_test':
        # For Chi-squared, we need the counts of 'Claim' and 'NoClaim' policies
        claim_count_g1 = group1_df['ClaimCount'].sum()
        no_claim_count_g1 = group1_df['TotalPolicies'].sum() - claim_count_g1

        claim_count_g2 = group2_df['ClaimCount'].sum()
        no_claim_count_g2 = group2_df['TotalPolicies'].sum() - claim_count_g2

        observed = pd.DataFrame({
            'Claim': [claim_count_g1, claim_count_g2],
            'NoClaim': [no_claim_count_g1, no_claim_count_g2]
        }, index=['Group1', 'Group2'])

        # Chi-squared test is unreliable with small expected frequencies (typically < 5)
        # Check if any cell in the observed table is very small or negative
        if observed.min().min() < 5 or no_claim_count_g1 < 0 or no_claim_count_g2 < 0:
            print(f"Warning: Chi-squared test for '{metric_name}' between groups may be unreliable due to low counts in contingency table:\n{observed}\nReturning NaN.")
            return np.nan

        chi2_statistic, p_value, _, _ = stats.chi2_contingency(observed)
        return p_value
    else:
        raise ValueError("Invalid test_type. Choose 't_test' or 'chi2_test'.")


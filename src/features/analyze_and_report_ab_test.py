import logging
import pandas as pd
from src import perform_hypothesis_test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_and_report_ab_test(aggregated_df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Performs A/B hypothesis testing for defined null hypotheses and reports the findings.

    Args:
        aggregated_df (pd.DataFrame): DataFrame with pre-calculated aggregated metrics.
                                     This DataFrame should contain the specific data
                                     you want to analyze (e.g., the tail data).
        alpha (float): Significance level for hypothesis testing (default is 0.05).
    """
    # Validate required columns
    required_columns = ['Province', 'PostalCode', 'Gender', 'ClaimFrequency', 'ClaimSeverity', 'Margin']
    missing_columns = [col for col in required_columns if col not in aggregated_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in DataFrame: {missing_columns}")
        return

    print("--- A/B Hypothesis Testing Analysis Report ---")
    print(f"Significance Level (alpha): {alpha}\n")

    # --- Province Comparison ---
    print("## 1. Province Risk and Profit Differences")
    provinces = aggregated_df['Province'].unique()
    if len(provinces) < 2:
        print("Not enough unique provinces in the provided data to perform comparison.\n")
    else:
        prov1_name = provinces[0]
        prov2_name = provinces[1]
        province_a = aggregated_df[aggregated_df['Province'] == prov1_name]
        province_b = aggregated_df[aggregated_df['Province'] == prov2_name]

        if province_a.empty or province_b.empty:
            print(f"Not enough data to compare provinces ({prov1_name} or {prov2_name} missing after filtering).\n")
        else:
            # H₀: No risk difference (Claim Frequency) across provinces
            p_value_prov_freq = perform_hypothesis_test(province_a, province_b, 'ClaimFrequency', 'chi2_test')
            if pd.isna(p_value_prov_freq):
                print(f"H₀: No risk difference (Claim Frequency) across provinces ({prov1_name} vs {prov2_name}): Test could not be performed or is unreliable (p = {p_value_prov_freq}). Need more data.")
            elif p_value_prov_freq < alpha:
                print(f"H₀: No risk difference (Claim Frequency) across provinces ({prov1_name} vs {prov2_name}): **REJECTED** (p = {p_value_prov_freq:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Frequency between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Freq {prov1_name}: {province_a['ClaimFrequency'].mean():.4f}, Mean Freq {prov2_name}: {province_b['ClaimFrequency'].mean():.4f}")
            else:
                print(f"H₀: No risk difference (Claim Frequency) across provinces ({prov1_name} vs {prov2_name}): **FAILED TO REJECT** (p = {p_value_prov_freq:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Frequency was found between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Freq {prov1_name}: {province_a['ClaimFrequency'].mean():.4f}, Mean Freq {prov2_name}: {province_b['ClaimFrequency'].mean():.4f}")

            # H₀: No risk difference (Claim Severity) across provinces
            p_value_prov_sev = perform_hypothesis_test(province_a, province_b, 'ClaimSeverity', 't_test')
            if pd.isna(p_value_prov_sev):
                print(f"H₀: No risk difference (Claim Severity) across provinces ({prov1_name} vs {prov2_name}): Test could not be performed (p = {p_value_prov_sev}).")
            elif p_value_prov_sev < alpha:
                print(f"H₀: No risk difference (Claim Severity) across provinces ({prov1_name} vs {prov2_name}): **REJECTED** (p = {p_value_prov_sev:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Severity between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Severity {prov1_name}: {province_a['ClaimSeverity'].mean():.2f}, Mean Severity {prov2_name}: {province_b['ClaimSeverity'].mean():.2f}")
            else:
                print(f"H₀: No risk difference (Claim Severity) across provinces ({prov1_name} vs {prov2_name}): **FAILED TO REJECT** (p = {p_value_prov_sev:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Severity was found between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Severity {prov1_name}: {province_a['ClaimSeverity'].mean():.2f}, Mean Severity {prov2_name}: {province_b['ClaimSeverity'].mean():.2f}")

            # H₀: No margin difference across provinces
            p_value_prov_margin = perform_hypothesis_test(province_a, province_b, 'Margin', 't_test')
            if pd.isna(p_value_prov_margin):
                print(f"H₀: No margin difference across provinces ({prov1_name} vs {prov2_name}): Test could not be performed (p = {p_value_prov_margin}).")
            elif p_value_prov_margin < alpha:
                print(f"H₀: No margin difference across provinces ({prov1_name} vs {prov2_name}): **REJECTED** (p = {p_value_prov_margin:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Margin between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Margin {prov1_name}: {province_a['Margin'].mean():.2f}, Mean Margin {prov2_name}: {province_b['Margin'].mean():.2f}")
            else:
                print(f"H₀: No margin difference across provinces ({prov1_name} vs {prov2_name}): **FAILED TO REJECT** (p = {p_value_prov_margin:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Margin was found between Province {prov1_name} and Province {prov2_name}.")
                print(f"   Mean Margin {prov1_name}: {province_a['Margin'].mean():.2f}, Mean Margin {prov2_name}: {province_b['Margin'].mean():.2f}")
    print("\n")

    # --- Zip Code Comparison ---
    print("## 2. Zip Code Risk and Profit Differences")
    zip_codes = aggregated_df['PostalCode'].unique()
    if len(zip_codes) < 2:
        print("Not enough unique zip codes in the provided data to perform comparison.\n")
    else:
        zip1_name = zip_codes[2]
        zip2_name = zip_codes[10]
        zip_1 = aggregated_df[aggregated_df['PostalCode'] == zip1_name]
        zip_2 = aggregated_df[aggregated_df['PostalCode'] == zip2_name]

        if zip_1.empty or zip_2.empty:
            print(f"Not enough data to compare zip codes ({zip1_name} or {zip2_name} missing after filtering).\n")
        else:
            # H₀: No risk difference (Claim Frequency) between zip codes
            p_value_zip_freq = perform_hypothesis_test(zip_1, zip_2, 'ClaimFrequency', 'chi2_test')
            if pd.isna(p_value_zip_freq):
                print(f"H₀: No risk difference (Claim Frequency) between zip codes ({zip1_name} vs {zip2_name}): Test could not be performed or is unreliable (p = {p_value_zip_freq}). Need more data.")
            elif p_value_zip_freq < alpha:
                print(f"H₀: No risk difference (Claim Frequency) between zip codes ({zip1_name} vs {zip2_name}): **REJECTED** (p = {p_value_zip_freq:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Frequency between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Freq {zip1_name}: {zip_1['ClaimFrequency'].mean():.4f}, Mean Freq {zip2_name}: {zip_2['ClaimFrequency'].mean():.4f}")
            else:
                print(f"H₀: No risk difference (Claim Frequency) between zip codes ({zip1_name} vs {zip2_name}): **FAILED TO REJECT** (p = {p_value_zip_freq:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Frequency was found between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Freq {zip1_name}: {zip_1['ClaimFrequency'].mean():.4f}, Mean Freq {zip2_name}: {zip_2['ClaimFrequency'].mean():.4f}")

            # H₀: No risk difference (Claim Severity) between zip codes
            p_value_zip_sev = perform_hypothesis_test(zip_1, zip_2, 'ClaimSeverity', 't_test')
            if pd.isna(p_value_zip_sev):
                print(f"H₀: No risk difference (Claim Severity) between zip codes ({zip1_name} vs {zip2_name}): Test could not be performed (p = {p_value_zip_sev}).")
            elif p_value_zip_sev < alpha:
                print(f"H₀: No risk difference (Claim Severity) between zip codes ({zip1_name} vs {zip2_name}): **REJECTED** (p = {p_value_zip_sev:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Severity between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Severity {zip1_name}: {zip_1['ClaimSeverity'].mean():.2f}, Mean Severity {zip2_name}: {zip_2['ClaimSeverity'].mean():.2f}")
            else:
                print(f"H₀: No risk difference (Claim Severity) between zip codes ({zip1_name} vs {zip2_name}): **FAILED TO REJECT** (p = {p_value_zip_sev:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Severity was found between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Severity {zip1_name}: {zip_1['ClaimSeverity'].mean():.2f}, Mean Severity {zip2_name}: {zip_2['ClaimSeverity'].mean():.2f}")

            # H₀: No margin difference between zip codes
            p_value_zip_margin = perform_hypothesis_test(zip_1, zip_2, 'Margin', 't_test')
            if pd.isna(p_value_zip_margin):
                print(f"H₀: No margin difference between zip codes ({zip1_name} vs {zip2_name}): Test could not be performed (p = {p_value_zip_margin}).")
            elif p_value_zip_margin < alpha:
                print(f"H₀: No margin difference between zip codes ({zip1_name} vs {zip2_name}): **REJECTED** (p = {p_value_zip_margin:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Margin between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Margin {zip1_name}: {zip_1['Margin'].mean():.2f}, Mean Margin {zip2_name}: {zip_2['Margin'].mean():.2f}")
            else:
                print(f"H₀: No margin difference between zip codes ({zip1_name} vs {zip2_name}): **FAILED TO REJECT** (p = {p_value_zip_margin:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Margin was found between Zip Code {zip1_name} and Zip Code {zip2_name}.")
                print(f"   Mean Margin {zip1_name}: {zip_1['Margin'].mean():.2f}, Mean Margin {zip2_name}: {zip_2['Margin'].mean():.2f}")
    print("\n")

    # --- Gender Comparison ---
    print("## 3. Gender Risk Differences")
    genders = aggregated_df['Gender'].unique()
    if len(genders) < 2:
        print("Not enough unique genders in the provided data to perform comparison.\n")
    else:
        gender1_name = genders[0]
        gender2_name = genders[1]
        gender1_df = aggregated_df[aggregated_df['Gender'] == gender1_name]
        gender2_df = aggregated_df[aggregated_df['Gender'] == gender2_name]

        if gender1_df.empty or gender2_df.empty:
            print(f"Not enough data to compare genders ({gender1_name} or {gender2_name} missing after filtering).\n")
        else:
            # H₀: No risk difference (Claim Frequency) between genders
            p_value_gender_freq = perform_hypothesis_test(gender1_df, gender2_df, 'ClaimFrequency', 'chi2_test')
            if pd.isna(p_value_gender_freq):
                print(f"H₀: No risk difference (Claim Frequency) between {gender1_name} and {gender2_name}: Test could not be performed or is unreliable (p = {p_value_gender_freq}). Need more data.")
            elif p_value_gender_freq < alpha:
                print(f"H₀: No risk difference (Claim Frequency) between {gender1_name} and {gender2_name}: **REJECTED** (p = {p_value_gender_freq:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Frequency between {gender1_name} and {gender2_name}.")
                print(f"   Mean Freq {gender1_name}: {gender1_df['ClaimFrequency'].mean():.4f}, Mean Freq {gender2_name}: {gender2_df['ClaimFrequency'].mean():.4f}")
            else:
                print(f"H₀: No risk difference (Claim Frequency) between {gender1_name} and {gender2_name}: **FAILED TO REJECT** (p = {p_value_gender_freq:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Frequency was found between {gender1_name} and {gender2_name}.")
                print(f"   Mean Freq {gender1_name}: {gender1_df['ClaimFrequency'].mean():.4f}, Mean Freq {gender2_name}: {gender2_df['ClaimFrequency'].mean():.4f}")

            # H₀: No risk difference (Claim Severity) between genders
            p_value_gender_sev = perform_hypothesis_test(gender1_df, gender2_df, 'ClaimSeverity', 't_test')
            if pd.isna(p_value_gender_sev):
                print(f"H₀: No risk difference (Claim Severity) between {gender1_name} and {gender2_name}: Test could not be performed (p = {p_value_gender_sev}).")
            elif p_value_gender_sev < alpha:
                print(f"H₀: No risk difference (Claim Severity) between {gender1_name} and {gender2_name}: **REJECTED** (p = {p_value_gender_sev:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Claim Severity between {gender1_name} and {gender2_name}.")
                print(f"   Mean Severity {gender1_name}: {gender1_df['ClaimSeverity'].mean():.2f}, Mean Severity {gender2_name}: {gender2_df['ClaimSeverity'].mean():.2f}")
            else:
                print(f"H₀: No risk difference (Claim Severity) between {gender1_name} and {gender2_name}: **FAILED TO REJECT** (p = {p_value_gender_sev:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Claim Severity was found between {gender1_name} and {gender2_name}.")
                print(f"   Mean Severity {gender1_name}: {gender1_df['ClaimSeverity'].mean():.2f}, Mean Severity {gender2_name}: {gender2_df['ClaimSeverity'].mean():.2f}")

            # H₀: No margin difference between genders
            p_value_gender_margin = perform_hypothesis_test(gender1_df, gender2_df, 'Margin', 't_test')
            if pd.isna(p_value_gender_margin):
                print(f"H₀: No margin difference between {gender1_name} and {gender2_name}: Test could not be performed (p = {p_value_gender_margin}).")
            elif p_value_gender_margin < alpha:
                print(f"H₀: No margin difference between {gender1_name} and {gender2_name}: **REJECTED** (p = {p_value_gender_margin:.4f}).")
                print(f"   Interpretation: There is a statistically significant difference in Margin between {gender1_name} and {gender2_name}.")
                print(f"   Mean Margin {gender1_name}: {gender1_df['Margin'].mean():.2f}, Mean Margin {gender2_name}: {gender2_df['Margin'].mean():.2f}")
            else:
                print(f"H₀: No margin difference between {gender1_name} and {gender2_name}: **FAILED TO REJECT** (p = {p_value_gender_margin:.4f}).")
                print(f"   Interpretation: No statistically significant difference in Margin was found between {gender1_name} and {gender2_name}.")
                print(f"   Mean Margin {gender1_name}: {gender1_df['Margin'].mean():.2f}, Mean Margin {gender2_name}: {gender2_df['Margin'].mean():.2f}")

    print("\n--- End of Report ---\n")
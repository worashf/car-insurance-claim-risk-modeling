

# 🚗 Car Insurance Claim Risk Modeling

## 📘 Project Overview

This project is a learning initiative aimed at developing cutting-edge risk and predictive analytics in the car insurance industry, with a focus on marketing and planning in South Africa.

As a **Marketing Analytics Engineer**, your objectives are to:

* Analyze historical insurance claim data.
* Identify “low-risk” customers for potential premium reductions.
* Optimize marketing strategies using data insights.
* Build reproducible, auditable pipelines using **DVC** to ensure compliance with financial regulations.

> 📁 **Repository**: `car-insurance-claim-risk-modeling`

---

## 🎯 Objectives

* 🔍 Perform **Exploratory Data Analysis (EDA)** to uncover patterns in risk and profitability.

* 📊 Conduct **A/B Hypothesis Testing** to evaluate differences across provinces, zip codes, and demographics.

* 📚 Learn and apply key **insurance terminology** for domain-specific accuracy.

* 🤖 Build statistical and machine learning models to predict:

  * Total insurance claims (**severity**).
  * Probability of claims occurring.
  * Optimal premium values (**dynamic pricing**).

* 🔁 Use **DVC** to manage datasets and ensure **reproducibility** and **auditability**.

---

## ✅ Tasks and Deliverables

### **Task 1: Git & GitHub Setup**

**Goals:**

* Initialize a Git repo with a clear `README.md`.
* Create a branch `task-1` and commit at least 3× per day with descriptive messages.
* Set up **GitHub Actions** for CI/CD.

**KPIs:**

* Working development environment.
* Proper Git/GitHub usage and clean commit history.

---

### **Task 1.2: EDA & Statistical Analysis**

#### ✳️ Data Understanding

* Assess data structure, quality, and financial fields.

#### ✳️ EDA Components

* **Descriptive Stats**: Summarize `TotalPremium`, `TotalClaims`, etc.
* **Data Types**: Format dates, categoricals, numerics.
* **Missing Values**: Identify and handle appropriately.
* **Univariate Analysis**: Histograms and bar plots.
* **Bivariate Analysis**: Correlation and scatter plots (`TotalPremium` vs `TotalClaims`, segmented by `ZipCode`).
* **Geographical Trends**: Compare across provinces/regions.
* **Outliers**: Use box plots.
* **Insightful Visualizations**: Create 3 meaningful, well-explained plots.

#### 📌 Guiding Questions

* What’s the overall **Loss Ratio** (`TotalClaims / TotalPremium`)? How does it vary?
* Are there outliers in `TotalClaims` or `CustomValueEstimate`?
* Do claim trends change over time (18-month window)?
* Which vehicle makes/models have the highest or lowest claims?

**KPIs:**

* Effective EDA techniques.
* Statistical insights through visualization.
* Demonstrated understanding of data.

---

### **Task 2: DVC Setup**

**Steps:**

```bash
pip install dvc
dvc init
mkdir /path/to/local/storage
dvc remote add -d localstorage /path/to/local/storage
dvc add data/your-data-file.csv
git add your-data-file.csv.dvc .gitignore
git commit -m "Track data with DVC"
dvc push
```

**Goal**: Ensure **data versioning** for traceability and reproducibility.

**KPIs:**

* Successful DVC setup and data tracking.
* Reproducible data pipeline.

---

### **Task 3: Learn Insurance Terminology**

**Goal**: Gain proficiency in key insurance concepts.

**Steps:**

* Study **50 common insurance terms**, focusing on:

  * **Premium**: Amount paid for insurance.
  * **Claim**: Request for payment due to loss/damage.
  * **Loss Ratio**: `TotalClaims / TotalPremium`.
  * **Underwriting**: Risk evaluation for premium setting.
  * **Deductible**: Amount the policyholder pays before insurance kicks in.

* Apply terminology in:

  * EDA notebooks
  * Model documentation
  * Project reports

* Create a `glossary.md` file under `notebooks/` with at least 10 terms and project-specific examples.

**KPIs:**

* Accurate and consistent terminology usage.
* Glossary committed to the repository.
* Demonstrated understanding in analysis and modeling.

---

### **Task 4: A/B Hypothesis Testing**

**Objective**: Detect risk and profitability differences across key segments.

#### Hypotheses:

* H1: No risk difference across **provinces**.
* H2: No claim distribution difference across **zip codes**.
* H3: No **profit margin** difference across zip codes.
* H4: No gender-based risk differences.

**Steps:**

1. **Prepare data** from Task 1.2 with relevant features.

2. Apply:

   * **T-tests/ANOVA** for continuous variables (e.g., `TotalClaims`).
   * **Chi-square tests** for categorical variables (e.g., gender).
   * **Bonferroni correction** to adjust for multiple comparisons.

3. Create visualizations:

   * Box plots, bar charts, heatmaps.

4. Document results in `notebooks/hypothesis_testing.ipynb`.

**Deliverables:**

* Jupyter notebook with:

  * Test statistics and p-values.
  * Null hypothesis outcomes.
  * Visual insights.
* Actionable marketing strategy inputs.

**KPIs:**

* Correct use of statistical tests.
* Clear visualizations and insights.
* Strategic relevance of findings.

---

### **Task 5: Predictive Modeling for Dynamic Risk-Based Pricing**

#### Goals:

Build three models:

1. **Claim Severity Model** (regression on `TotalClaims` > 0)
2. **Claim Probability Model** (classification: claim vs no-claim)
3. **Premium Optimization** (pricing strategy)

---

#### 1. Data Preparation:

* **Imputation**: Fill missing values appropriately.
* **Feature Engineering**:

  * `VehicleAge = CurrentYear - RegistrationYear`
  * `LossRatio = TotalClaims / TotalPremium`
* **Encoding**:

  * One-hot for categorical (Gender, Province, etc.)
  * Label encoding as needed
* **Train-Test Split** (e.g., 80/20 stratified for classification)

---

#### 2. Modeling Techniques:

* **Linear Regression** (baseline severity model)
* **Decision Trees**
* **Random Forests**
* **XGBoost** (handles class imbalance via `scale_pos_weight`)

---

#### 3. Implementation:

* Use a `DynamicRiskPricing` class (e.g., in `scripts/dynamic_risk_pricing.py`)
* Predict severity and probability separately
* Combine for premium:

  ```python
  Premium = (Predicted_Probability * Predicted_Severity) + Expense_Loading + Profit_Margin
  ```

---

#### 4. Evaluation:

| Model       | Metric                                   |
| ----------- | ---------------------------------------- |
| Severity    | RMSE, R²                                 |
| Probability | Accuracy, Precision, Recall, F1, ROC AUC |
| Premium     | Profitability, Business alignment        |

---

**Deliverables:**

* Notebook: `notebooks/modeling.ipynb`
* Python class: `scripts/dynamic_risk_pricing.py`
* Visuals: Feature importances, confusion matrix, premium histograms

**KPIs:**

* RMSE < threshold, R² > 0.7
* F1 > 0.5, ROC AUC > 0.8
* Business-aligned premium predictions
* Reproducible pipeline using DVC

---

## 🗂️ Project Structure

```
car-insurance-claim-risk-modeling/
├── data/                      # Raw and processed data (DVC-tracked)
├── notebooks/
│   ├── glossary.md            # Glossary of insurance terms
│   ├── hypothesis_testing.ipynb
│   └── modeling.ipynb
├── scripts/
│   └── dynamic_risk_pricing.py
├── .dvc/                      # DVC configuration
├── .github/workflows/         # CI/CD setup
├── README.md
├── requirements.txt
└── environment.yml
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone git@github.com:worashf/car-insurance-claim-risk-modeling.git
cd car-insurance-claim-risk-modeling

# Install dependencies
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate car-insurance-predictive-model

# Initialize DVC and pull data
dvc init
dvc remote add -d localstorage /path/to/local/storage
dvc pull

# Launch
jupyter lab
```

---

## 🤝 Contributing

This is a collaborative learning project.

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Commit with clear messages.

3. Push and open a Pull Request.

---

## 📜 License

This repository is intended for **educational purposes only**.

---

Let me know if you want this in `README.md` format or need a PDF export!



🎯 Objectives

🔍 Perform Exploratory Data Analysis (EDA) to identify patterns in risk and profitability.
📊 Conduct A/B Hypothesis Testing to evaluate differences across provinces, zip codes, and demographics.
📚 Learn and apply key insurance terminology to ensure domain-specific accuracy.
🤖 Build statistical and machine learning models to predict:
Total insurance claims (severity).
Probability of claims occurring.
Optimal premium values for risk-based pricing.


🔁 Use Data Version Control (DVC) to manage datasets for reproducibility and auditability.


✅ Tasks and Deliverables
Task 1: Git & GitHub Setup

Initialize a Git repo with a clear README.md.
Create a branch named task-1 and commit at least 3x/day with descriptive messages.
Set up GitHub Actions for CI/CD.

KPIs:

Working dev environment.
Proper Git/GitHub usage and commit hygiene.


Task 1.2: EDA & Statistical Analysis
✳️ Data Understanding

Assess data structure, quality, and financial fields.

✳️ EDA Components

Summarization: Descriptive stats for TotalPremium, TotalClaims, etc.
Data Types: Ensure proper formatting for dates, categoricals, numerics.
Missing Values: Identify and handle.
Univariate Analysis: Histograms and bar plots.
Bivariate Analysis: Scatter plots, correlations (TotalPremium vs TotalClaims by ZipCode).
Geographical Trends: Compare variables across provinces/regions.
Outlier Detection: Use box plots.
Insightful Plots: Create 3 well-explained, meaningful visualizations.

📌 Guiding Questions

What’s the overall Loss Ratio (TotalClaims / TotalPremium)? How does it vary?
Are there outliers in TotalClaims or CustomValueEstimate?
Do claim trends vary over time (18-month window)?
Which vehicle makes/models have the highest or lowest claims?

KPIs:

Use of effective EDA techniques.
Demonstrated statistical reasoning via plots and distributions.
Proactive learning and reference sharing.


Task 2: DVC Setup
Steps:
pip install dvc
dvc init
mkdir /path/to/your/local/storage
dvc remote add -d localstorage /path/to/your/local/storage
dvc add data/your-data-file.csv
git add your-data-file.csv.dvc .gitignore
git commit -m "Track data with DVC"
dvc push

Goal: Version data inputs to ensure reproducibility, traceability, and auditability.
KPIs:

Successful DVC initialization and data tracking.
Data pipeline reproducibility verified.


Task 3: Learn Insurance Terminology
Objective:
Gain proficiency in insurance domain knowledge to ensure accurate application of concepts in data analysis and modeling.
Steps:

Study the 50 Common Insurance Terms.
Focus on terms relevant to the project, such as:
Premium: The amount paid for insurance coverage.
Claim: A request for payment due to loss or damage.
Loss Ratio: The ratio of TotalClaims to TotalPremium, indicating profitability.
Underwriting: The process of evaluating risk to determine premiums.
Deductible: The amount a policyholder pays before insurance covers a claim.


Apply terminology in EDA, modeling, and documentation (e.g., use “Loss Ratio” correctly in README.md and notebooks).
Create a glossary in the notebooks/ directory (e.g., glossary.md) summarizing key terms and their relevance to the project.

Deliverables:

A documented glossary of at least 10 relevant terms with definitions and examples in the context of the project.
Evidence of terminology usage in EDA reports, code comments, or model documentation.

KPIs:

Accurate use of insurance terminology in project deliverables.
Glossary file committed to the repository.
Demonstrated understanding in analysis and modeling outputs.


Task 4: A/B Hypothesis Testing
Objective:
Evaluate risk and profitability differences across key segments to inform marketing strategies.
Hypotheses to Test:

H1: There are no risk differences across provinces (e.g., claim frequency or severity).
H2: There are no risk differences across zip codes (e.g., TotalClaims distribution).
H3: There are no significant profit margin differences across zip codes (e.g., TotalPremium - TotalClaims).
H4: There are no significant gender-based risk differences (e.g., claim probability by gender).

Steps:

Data Preparation: Use preprocessed data from Task 1.2 (EDA) with features like Province, ZipCode, Gender, TotalClaims, and TotalPremium.
Statistical Tests:
Use t-tests or ANOVA for continuous variables (e.g., TotalClaims across provinces).
Use chi-square tests for categorical variables (e.g., claim occurrence by gender).
Apply Bonferroni correction for multiple comparisons to control Type I errors.


Visualizations: Create box plots, bar charts, or heatmaps to visualize differences (e.g., LossRatio by province).
Interpretation: Accept or reject null hypotheses based on p-values (threshold: 0.05) and document findings in a notebook (e.g., notebooks/hypothesis_testing.ipynb).

Deliverables:

A Jupyter notebook summarizing test results, including:
Test statistics, p-values, and conclusions for each hypothesis.
Visualizations (e.g., box plots for TotalClaims by province).


Key findings integrated into the marketing strategy (e.g., target low-risk provinces).

KPIs:

Correct application of statistical tests.
Clear visualizations and interpretations.
Actionable insights for marketing (e.g., low-risk segments identified).


Task 5: Build and Evaluate Predictive Models for Dynamic Risk-Based Pricing
Objective:
Develop and evaluate predictive models to form the core of a dynamic, risk-based pricing system for car insurance, enabling accurate claim severity prediction, claim probability estimation, and premium optimization.
Modeling Goals:

Claim Severity Prediction (Risk Model):
Build a regression model to predict TotalClaims for policies with claims (where TotalClaims > 0).
Target Variable: TotalClaims (on the subset of data where claims > 0).
Evaluation Metrics: Root Mean Squared Error (RMSE) to penalize large prediction errors, and R-squared to measure explained variance.


Claim Probability Prediction:
Build a binary classification model to predict the probability of a claim occurring (HasClaim: 1 if TotalClaims > 0, else 0).
Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC (with focus on handling class imbalance).


Premium Optimization (Pricing Framework):
Develop a model to predict optimal premiums, combining predicted claim probability and severity.
Formula: Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin.
Move beyond naive prediction of CalculatedPremiumPerTerm to a business-driven approach incorporating risk and cost factors.



Tasks:

Data Preparation:
Handling Missing Data: Impute missing values (e.g., median for numeric, mode for categorical) or remove rows/columns based on missingness extent.
Feature Engineering: Create features like VehicleAge (from RegistrationYear), LossRatio (TotalClaims / TotalPremium), or risk indicators (e.g., high-value vehicle flags).
Encoding Categorical Data: Use one-hot encoding for categorical features (e.g., Gender, Province, VehicleType) or label encoding where appropriate.
Train-Test Split: Split data into 80:20 or 70:30 train-test sets, ensuring stratification for classification tasks.


Modeling Techniques:
Linear Regression: Baseline model for claim severity prediction.
Decision Trees: Simple tree-based model for both regression and classification.
Random Forests: Ensemble model for improved performance and robustness.
Gradient Boosting Machines (GBMs):
XGBoost: High-performance model for both severity (regression) and probability (classification), with support for class imbalance handling (e.g., scale_pos_weight).




Implementation:
Use a class like DynamicRiskPricing (in scripts/dynamic_risk_pricing.py) to encapsulate preprocessing, modeling, and premium calculation.
Train separate models for severity (TotalClaims) and probability (HasClaim).
Combine outputs for premium optimization using the formula above, with configurable Expense Loading (e.g., 10%) and Profit Margin (e.g., 15%).


Evaluation:
For severity: Compute RMSE and R-squared on the test set.
For probability: Compute accuracy, precision, recall, F1, and ROC AUC, addressing class imbalance (e.g., using scale_pos_weight or SMOTE).
For premiums: Validate predicted premiums against business constraints (e.g., profitability, competitiveness).



Deliverables:

A Jupyter notebook documenting:
Data preparation steps.
Model training and evaluation for severity and probability.
Premium optimization results with sample predictions.


A Python script  implementing the DynamicRiskPricing class.
Visualizations (e.g., feature importance plots, confusion matrices, premium distributions).

KPIs:

Model performance: RMSE < [threshold], R-squared > 0.7 for severity; F1 > 0.5, ROC AUC > 0.8 for probability.
Identification of low-risk customer segments for marketing.
Reproducible pipeline with DVC-tracked data and models.
Actionable premium predictions aligned with business goals (e.g., profitability).


🗂️ Project Structure
car-insurance-claim-risk-modeling/
├── data/                    # Raw and processed datasets (tracked by DVC)
├── notebooks/               # EDA, hypothesis testing, and modeling notebooks
│   
├── scripts/                 # Python scripts for pipeline steps
│---src
├── .dvc/                    # DVC configuration
├── .github/workflows/       # GitHub Actions CI/CD pipelines
├── README.md                # Project documentation
├── requirements.txt         # Dependencies for pip-based installs
└── environment.yml          # Conda environment config


🚀 Getting Started
# Clone the repository
git clone git@github.com:worashf/car-insurance-claim-risk-modeling.git
cd car-insurance-claim-risk-modeling

# Install dependencies (via pip or conda)
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate car-insurance-predictive-model

# Initialize and configure DVC
dvc init
dvc remote add -d localstorage /path/to/your/local/storage
dvc pull

# Start analyzing!
jupyter lab


🤝 Contributing
This is a collaborative learning project. To contribute:

Create a branch:git checkout -b feature/your-feature-name


Commit with clear messages.
Push and open a pull request.


📜 License
This repository is intended for learning purposes only

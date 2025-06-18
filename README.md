
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

4. Document results in `notebooks/`.

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

* Notebooks
* Python methods and class 
* Visuals

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
│
├── scripts/
]-- src
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

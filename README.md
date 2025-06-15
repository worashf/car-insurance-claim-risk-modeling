

# 🚗 Car Insurance Claim Risk Modeling

## 📘 Project Overview

This project is a learning initiative to develop cutting-edge risk and predictive analytics in the car insurance industry, focusing on marketing and planning in South Africa.

As a **Marketing Analytics Engineer**, your objectives are to:

- Analyze historical insurance claim data.
- Identify “low-risk” customers for potential premium reductions.
- Optimize marketing strategies using data insights.
- Build reproducible, auditable pipelines using **DVC** for compliance with financial regulations.

> 📁 Repository: [car-insurance-claim-risk-modeling](git@github.com:worashf/car-insurance-claim-risk-modeling.git)

---

## 🎯 Objectives

- 🔍 Perform Exploratory Data Analysis (EDA) to identify patterns in risk and profitability.
- 📊 Conduct A/B Hypothesis Testing to evaluate differences across provinces, zip codes, and demographics.
- 🤖 Build statistical and machine learning models to predict:
  - Total insurance claims.
  - Optimal premium values.
- 🔁 Use **Data Version Control (DVC)** to manage datasets for reproducibility and auditability.

---

## ✅ Tasks and Deliverables

### Task 1: Git & GitHub Setup
- Initialize a Git repo with a clear `README.md`.
- Create a branch named `task-1` and commit at least 3x/day with descriptive messages.
- Set up **GitHub Actions** for CI/CD.

**KPIs:**
- Working dev environment.
- Proper Git/GitHub usage and commit hygiene.

---

### Task 1.2: EDA & Statistical Analysis

#### ✳️ Data Understanding
- Assess data structure, quality, and financial fields.

#### ✳️ EDA Components
- **Summarization:** Descriptive stats for `TotalPremium`, `TotalClaims`, etc.
- **Data Types:** Ensure proper formatting for dates, categoricals, numerics.
- **Missing Values:** Identify and handle.
- **Univariate Analysis:** Histograms and bar plots.
- **Bivariate Analysis:** Scatter plots, correlations (`TotalPremium` vs `TotalClaims` by ZipCode).
- **Geographical Trends:** Compare variables across provinces/regions.
- **Outlier Detection:** Use box plots.
- **Insightful Plots:** Create 3 well-explained, meaningful visualizations.

#### 📌 Guiding Questions
- What’s the overall **Loss Ratio** (`TotalClaims / TotalPremium`)? How does it vary?
- Are there outliers in `TotalClaims` or `CustomValueEstimate`?
- Do claim trends vary over time (18-month window)?
- Which vehicle makes/models have the highest or lowest claims?

**KPIs:**
- Use of effective EDA techniques.
- Demonstrated statistical reasoning via plots and distributions.
- Proactive learning and reference sharing.

---

### Task 2: DVC Setup

#### Steps:
```bash
pip install dvc
dvc init
mkdir /path/to/your/local/storage
dvc remote add -d localstorage /path/to/your/local/storage
dvc add data/your-data-file.csv
git add your-data-file.csv.dvc .gitignore
git commit -m "Track data with DVC"
dvc push
````

**Goal:** Version data inputs to ensure reproducibility, traceability, and auditability.

---

### Task 3: Learn Insurance Terminology

Study and apply key terms from [50 Common Insurance Terms](https://www.cornerstoneins.ca/blog/50-common-insurance-terms-and-what-they-mean/).

---

### Task 4: A/B Hypothesis Testing

#### Hypotheses to Test:

* No risk differences across **provinces**.
* No risk differences across **zip codes**.
* No significant **profit margin** differences across zip codes.
* No significant **gender-based risk** differences.

Use appropriate tests (e.g., t-test, ANOVA, chi-square) and visualizations to accept or reject the null hypotheses.

---

### Task 5: Modeling

#### Statistical Models:

* Fit **linear regression** models to predict `TotalClaims` by ZipCode.

#### Machine Learning:

Predict **optimal premiums** using:

* Car features (make, model, value)
* Owner features (gender, age)
* Location (province, zip code)
* Other relevant features from EDA

**Goal:** Identify low-risk segments for marketing strategy optimization.

---

## 🗂️ Project Structure

```
car-insurance-claim-risk-modeling/
├── data/                    # Raw and processed datasets (tracked by DVC)
├── notebooks/               # EDA, modeling, and hypothesis testing notebooks
├── scripts/                 # Python scripts for pipeline steps
├── .dvc/                    # DVC configuration
├── .github/workflows/       # GitHub Actions CI/CD pipelines
├── README.md                # Project documentation
├── requirements.txt         # Dependencies for pip-based installs
└── environment.yml          # Conda environment config
```

---

## 🚀 Getting Started

```bash
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
```

---

## 🤝 Contributing

This is a collaborative learning project. To contribute:

1. Create a branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit with clear messages.
3. Push and open a pull request.

---

## 📜 License

This repository is intended **for learning purposes only**
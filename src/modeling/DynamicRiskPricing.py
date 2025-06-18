# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicRiskPricing:
    """
    A system to build and evaluate models for risk-based pricing in car insurance.
    Tasks:
    - Claim Severity Prediction (Regression)
    - Claim Probability Prediction (Classification)
    - Risk-Based Premium Optimization
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.severity_model = None
        self.probability_model = None

        # Define numeric and categorical features
        self.numeric_features = ['cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CustomValueEstimate',
                                 'SumInsured', 'CapitalOutstanding', 'CalculatedPremiumPerTerm']
        self.categorical_features = ['Gender', 'Province', 'VehicleType', 'CoverType', 'make', 'Model',
                                      'bodytype', 'IsVATRegistered']

    def handle_missing_data(self):
        """Handle missing data in the dataset."""
        self.df.fillna({
            col: self.df[col].median() if self.df[col].dtype != 'object' else self.df[col].mode()[0]
            for col in self.df.columns
        }, inplace=True)
        logger.info("Handled missing data.")

    def feature_engineering(self):
        """Create new features like LossRatio and VehicleAge."""
        current_year = pd.Timestamp.now().year
        if 'RegistrationYear' in self.df.columns:
            self.df['VehicleAge'] = current_year - self.df['RegistrationYear']
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'].replace(0, 1e-6))
        logger.info("Feature engineering completed.")

    def prepare_data(self, target: str, claim_filter: bool = False):
        """
        Prepare data for modeling:
        - Filters data if claim_filter is True.
        - Encodes categorical features and scales numeric features.
        Returns: train-test splits and preprocessing pipeline.
        """
        data = self.df.copy()

        # Filter data for claims > 0 if required
        if claim_filter:
            data = data[data['TotalClaims'] > 0].copy()

        # Transform target
        if target == 'TotalClaims':
            data['LogTotalClaims'] = np.log1p(data['TotalClaims'])
            target = 'LogTotalClaims'
        elif target == 'HasClaim':
            data['HasClaim'] = (data['TotalClaims'] > 0).astype(int)

        # Define features
        numeric_features = [f for f in self.numeric_features if f in data.columns]
        categorical_features = [f for f in self.categorical_features if f in data.columns]

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Features and target
        X = data.drop(columns=[target, 'TotalClaims', 'TotalPremium', 'RegistrationYear'], errors='ignore')
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, preprocessor

    def train_severity_model(self):
        """Train a regression model to predict claim severity."""
        logger.info("Training claim severity model...")
        X_train, X_test, y_train, y_test, preprocessor = self.prepare_data(target='TotalClaims', claim_filter=True)

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Reverse log transform for evaluation
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        r2 = r2_score(y_test_actual, y_pred_actual)
        logger.info(f"Severity Model - RMSE: {rmse:.2f}, R2: {r2:.2f}")

        self.severity_model = model
        return {'RMSE': rmse, 'R2': r2}

    def train_probability_model(self):
        """Train a classification model to predict claim probability."""
        logger.info("Training claim probability model...")
        X_train, X_test, y_train, y_test, preprocessor = self.prepare_data(target='HasClaim', claim_filter=False)

        # Calculate class weights for imbalance handling
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=scale_pos_weight))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        logger.info(f"Probability Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")

        self.probability_model = model
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'ROC_AUC': roc_auc}

    def optimize_premiums(self, expense_load: float = 0.1, profit_margin: float = 0.15):
        """Calculate risk-based premiums."""
        if not self.severity_model or not self.probability_model:
            raise ValueError("Both severity and probability models must be trained first.")

        logger.info("Optimizing premiums...")
        X = self.df.drop(columns=['TotalClaims', 'TotalPremium', 'HasClaim', 'RegistrationYear'], errors='ignore')

        # Predict probabilities
        prob_preprocessor = self.probability_model.named_steps['preprocessor']
        X_processed_prob = prob_preprocessor.transform(X)
        claim_probabilities = self.probability_model.named_steps['classifier'].predict_proba(X_processed_prob)[:, 1]

        # Predict severities
        sev_preprocessor = self.severity_model.named_steps['preprocessor']
        X_processed_sev = sev_preprocessor.transform(X)
        claim_severities = np.expm1(self.severity_model.named_steps['regressor'].predict(X_processed_sev))

        # Calculate premiums
        expected_loss = claim_probabilities * claim_severities
        premiums = expected_loss * (1 + expense_load + profit_margin)
        return premiums
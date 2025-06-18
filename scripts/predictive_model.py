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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, ConfusionMatrixDisplay, roc_auc_score
import shap
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceModeling:
    """
    Comprehensive insurance predictive modeling system using actual column names
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_input_data()
        self.severity_model = None
        self.probability_model = None
        self.premium_model = None

        # Define all potential feature types, correctly classifying them.
        # Removed columns that are used for feature engineering (RegistrationYear, TotalPremium)
        # or that are inherently categorical but might have numeric-like values (e.g., Cylinders, PostalCode)
        # or are booleans.
        self.base_numeric_features = [
            'cubiccapacity',
            'kilowatts',
            'NumberOfDoors',
            'CustomValueEstimate',
            'SumInsured',
            'CapitalOutstanding',  # Added from the full index
            'CalculatedPremiumPerTerm'  # Added from the full index
        ]
        self.base_categorical_features = [
            'Gender',
            'Province',
            'PostalCode',  # Classified as categorical
            'VehicleType',
            'make',
            'Model',
            'bodytype',
            'CoverType',
            'IsVATRegistered',  # Classified as categorical
            'Citizenship',  # Classified as categorical
            'LegalType',  # Classified as categorical
            'Title',  # Classified as categorical
            'Language',  # Classified as categorical
            'Bank',  # Classified as categorical
            'AccountType',  # Classified as categorical
            'MaritalStatus',  # Classified as categorical
            'Country',  # Classified as categorical
            'MainCrestaZone',  # Classified as categorical
            'SubCrestaZone',  # Classified as categorical
            'ItemType',  # Classified as categorical
            'mmcode',  # Classified as categorical
            'Cylinders',  # Classified as categorical (even if numeric values, they represent categories)
            'VehicleIntroDate',  # Consider if this should be datetime and engineered, or just categorical
            'AlarmImmobiliser',  # Classified as categorical
            'TrackingDevice',  # Classified as categorical
            'NewVehicle',  # Classified as categorical
            'WrittenOff',  # Classified as categorical
            'Rebuilt',  # Classified as categorical
            'Converted',  # Classified as categorical
            'TermFrequency',  # Classified as categorical
            'CoverCategory',  # Classified as categorical
            'CoverGroup',  # Classified as categorical
            'Section',  # Classified as categorical
            'Product',  # Classified as categorical
            'StatutoryClass',  # Classified as categorical
            'StatutoryRiskType'  # Classified as categorical
        ]

        # Store the feature names after preprocessing for SHAP
        self.severity_feature_names_out = None
        self.probability_feature_names_out = None

    def _validate_input_data(self):
        """Validate that required columns exist in the input data"""
        required_columns = ['TotalClaims', 'TotalPremium', 'Gender',
                            'Province', 'VehicleType', 'CoverType']

        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

    def _prepare_data(self, target: str, claim_filter: bool = False) -> tuple:
        """
        Prepare data for modeling with proper preprocessing pipeline.
        Returns: (X_train_unprocessed, X_test_unprocessed, y_train, y_test),
                 preprocessor (ColumnTransformer definition),
                 original_feature_cols_for_transformer (list of column names that will go into preprocessor)
        """
        data = self.df.copy()

        current_year = pd.Timestamp.now().year
        if 'RegistrationYear' in data.columns:
            data['VehicleAge'] = current_year - data['RegistrationYear']

        if 'TotalClaims' in data.columns and 'TotalPremium' in data.columns:
            # Ensure TotalPremium is not zero to avoid division by zero
            data['LossRatio'] = data['TotalClaims'] / (data['TotalPremium'].replace(0, 1e-6))

        if claim_filter:
            data = data[data['TotalClaims'] > 0].copy()  # Ensure copy after filtering
            logger.info(f"Filtered for claims > 0. Remaining rows: {len(data)}")
            if data.empty:
                raise ValueError("Dataframe is empty after applying claim filter. Cannot proceed.")

        original_target = target  # Store original target name
        if target == 'TotalClaims':
            data['LogTotalClaims'] = np.log1p(data['TotalClaims'])
            target = 'LogTotalClaims'  # Update target for this function's scope
        elif target == 'HasClaim':
            data['HasClaim'] = (data['TotalClaims'] > 0).astype(int)
            # No change to `target` string for this case, it's already 'HasClaim'

        # Identify features that will go into the preprocessor, based on the full set of potential features
        # 'VehicleAge' and 'LossRatio' are included here if they were engineered
        final_numeric_features = [f for f in self.base_numeric_features + ['VehicleAge', 'LossRatio'] if
                                  f in data.columns]
        final_categorical_features = [f for f in self.base_categorical_features if f in data.columns]

        # Remove duplicates using dict.fromkeys (order-preserving for Python 3.7+)
        final_numeric_features = list(dict.fromkeys(final_numeric_features))
        final_categorical_features = list(dict.fromkeys(final_categorical_features))

        # Define preprocessing pipeline components
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sparse_output=False for SHAP
        ])

        # Build transformers list for ColumnTransformer
        transformers = []
        if final_numeric_features:
            transformers.append(('num', numeric_transformer, final_numeric_features))
        if final_categorical_features:
            transformers.append(('cat', categorical_transformer, final_categorical_features))

        if not transformers:
            raise ValueError(
                "No features available after selection for ColumnTransformer. Check input data and feature lists.")

        preprocessor_definition = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop other columns not explicitly listed to get a clean feature set
        )

        # Determine columns to drop from X to get the unprocessed features
        # These are the original columns that are either targets or used for feature engineering
        cols_to_drop_from_X = [original_target]
        if original_target == 'TotalClaims':
            cols_to_drop_from_X.append('LogTotalClaims')
        elif original_target == 'HasClaim':
            cols_to_drop_from_X.append('HasClaim')

        # Explicitly drop original 'RegistrationYear' and 'TotalClaims', 'TotalPremium' as they are now replaced by engineered features or are targets
        cols_to_drop_from_X.extend(['RegistrationYear', 'TotalClaims', 'TotalPremium'])
        # Filter to ensure column exists and is not the current `target` variable itself
        cols_to_drop_from_X = [col for col in cols_to_drop_from_X if col in data.columns and col != target]

        X = data.drop(columns=cols_to_drop_from_X, errors='ignore')
        y = data[target]

        # Filter X to only include columns the preprocessor will expect for its `fit`/`transform`
        cols_for_preprocessor = final_numeric_features + final_categorical_features
        X = X[cols_for_preprocessor].copy()  # IMPORTANT: .copy() to avoid SettingWithCopyWarning

        # *** NEW: Explicit type conversion and checks before passing to ColumnTransformer ***
        for col in final_numeric_features:
            if col in X.columns:
                # Attempt to convert to numeric, coercing errors to NaN
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Log any NaNs introduced by coercion (these will be imputed by SimpleImputer)
                if X[col].isnull().any():
                    logger.warning(f"Column '{col}' had non-numeric values coerced to NaN. These will be imputed.")

        for col in final_categorical_features:
            if col in X.columns:
                # Ensure categorical columns are treated as strings
                X[col] = X[col].astype(str)

        if X.empty or len(X) == 0:
            raise ValueError(
                "Feature DataFrame X is empty after data preparation and column dropping. Check filtering and feature selection.")

        return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor_definition, cols_for_preprocessor

    def _get_X_for_prediction(self, df_to_predict: pd.DataFrame, target_col_for_dropping: str,
                              apply_claim_filter: bool) -> tuple:
        """
        Helper to prepare X (unprocessed) and its column names for prediction,
        mimicking the feature engineering and selection of _prepare_data.
        Returns: X_unprocessed, cols_for_preprocessor_pred, original_indices
        """
        data_copy = df_to_predict.copy()
        current_year_ = pd.Timestamp.now().year

        if 'RegistrationYear' in data_copy.columns:
            data_copy['VehicleAge'] = current_year_ - data_copy['RegistrationYear']
        if 'TotalClaims' in data_copy.columns and 'TotalPremium' in data_copy.columns:
            data_copy['LossRatio'] = data_copy['TotalClaims'] / (data_copy['TotalPremium'].replace(0, 1e-6))

        if apply_claim_filter:
            data_copy = data_copy[data_copy['TotalClaims'] > 0].copy()
            if data_copy.empty:
                # Return empty DataFrame and empty feature list if filtered out
                return pd.DataFrame(), [], pd.Index([])  # Return empty Index as well

        # Identify features that will go into the preprocessor, based on the full set of potential features
        final_numeric_features_pred = [f for f in self.base_numeric_features + ['VehicleAge', 'LossRatio'] if
                                       f in data_copy.columns]
        final_categorical_features_pred = [f for f in self.base_categorical_features if f in data_copy.columns]

        final_numeric_features_pred = list(dict.fromkeys(final_numeric_features_pred))
        final_categorical_features_pred = list(dict.fromkeys(final_categorical_features_pred))

        cols_to_drop_ = [target_col_for_dropping]
        if target_col_for_dropping == 'TotalClaims':
            cols_to_drop_.append('LogTotalClaims')
        elif target_col_for_dropping == 'HasClaim':
            cols_to_drop_.append('HasClaim')

        cols_to_drop_.extend(['RegistrationYear', 'TotalClaims', 'TotalPremium'])
        cols_to_drop_ = [col for col in cols_to_drop_ if col in data_copy.columns and col != target_col_for_dropping]

        X_unprocessed_ = data_copy.drop(columns=cols_to_drop_, errors='ignore')

        # Filter X_unprocessed_ to only include columns the preprocessor will expect
        cols_for_preprocessor_pred = final_numeric_features_pred + final_categorical_features_pred
        X_unprocessed_ = X_unprocessed_[cols_for_preprocessor_pred].copy()  # IMPORTANT: .copy()

        # *** NEW: Explicit type conversion and checks for prediction data ***
        for col in final_numeric_features_pred:
            if col in X_unprocessed_.columns:
                X_unprocessed_[col] = pd.to_numeric(X_unprocessed_[col], errors='coerce')

        for col in final_categorical_features_pred:
            if col in X_unprocessed_.columns:
                X_unprocessed_[col] = X_unprocessed_[col].astype(str)

        return X_unprocessed_, cols_for_preprocessor_pred, data_copy.index  # Return original index of filtered data

    def train_severity_model(self):
        """Train model to predict claim severity (log(TotalClaims))"""
        logger.info("Training claim severity model")

        try:
            # _prepare_data returns (X_train_unprocessed, X_test_unprocessed, y_train, y_test),
            # preprocessor_def (ColumnTransformer object), and cols_for_preprocessor (list of feature names for CT)
            (X_train_unprocessed, X_test_unprocessed, y_train,
             y_test), preprocessor_def, cols_for_preprocessor = self._prepare_data(
                target='TotalClaims', claim_filter=True)

            model = Pipeline(steps=[
                ('preprocessor', preprocessor_def),  # preprocessor_def is the ColumnTransformer definition
                ('regressor', XGBRegressor(n_estimators=100, random_state=42))
            ])

            model.fit(X_train_unprocessed, y_train)  # Fit the entire pipeline on unprocessed X_train

            # Store the fitted preprocessor's feature names for SHAP
            self.severity_feature_names_out = model.named_steps['preprocessor'].get_feature_names_out(
                cols_for_preprocessor)

            # Evaluate the model on the transformed test set
            X_test_processed = model.named_steps['preprocessor'].transform(X_test_unprocessed)
            y_pred = model.named_steps['regressor'].predict(X_test_processed)  # Predict on processed X_test

            # Convert back from log-transformed for RMSE calculation
            rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
            r2 = r2_score(y_test, y_pred)
            logger.info(f"Severity model - RMSE: {rmse:.2f}, R2: {r2:.2f}")

            self.severity_model = model
            return {'RMSE': rmse, 'R2': r2}

        except Exception as e:
            logger.error(f"Error training severity model: {str(e)}")
            raise

    def train_probability_model(self):
        """Train model to predict claim probability with imbalance handling"""
        logger.info("Training claim probability model")

        try:
            # Prepare data - ensure we get the unprocessed version for proper weighting
            (X_train_unprocessed, X_test_unprocessed, y_train,
             y_test), preprocessor_def, cols_for_preprocessor = self._prepare_data(
                target='HasClaim', claim_filter=False)

            # Calculate class weights
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

            logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
            logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

            model = Pipeline(steps=[
                ('preprocessor', preprocessor_def),
                ('classifier', XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='aucpr',  # Better metric for imbalanced data
                    early_stopping_rounds=10
                ))
            ])

            # Fit with eval set for early stopping
            X_train_processed = model.named_steps['preprocessor'].fit_transform(X_train_unprocessed)
            model.named_steps['classifier'].fit(
                X_train_processed, y_train,
                eval_set=[(X_train_processed, y_train)],
                verbose=False
            )

            # Store feature names for SHAP
            self.probability_feature_names_out = model.named_steps['preprocessor'].get_feature_names_out(
                cols_for_preprocessor)

            # Comprehensive evaluation
            X_test_processed = model.named_steps['preprocessor'].transform(X_test_unprocessed)
            y_pred = model.named_steps['classifier'].predict(X_test_processed)
            y_proba = model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Log full metrics
            logger.info(f"Probability model metrics:")
            logger.info(f"- Accuracy: {accuracy:.4f}")
            logger.info(f"- F1: {f1:.4f}")
            logger.info(f"- Precision: {precision:.4f}")
            logger.info(f"- Recall: {recall:.4f}")
            logger.info(f"- ROC AUC: {roc_auc:.4f}")

            # Classification report
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix_probability.png")
            plt.close()
            logger.info("Saved confusion matrix plot")

            self.probability_model = model
            return {
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'ROC_AUC': roc_auc
            }

        except Exception as e:
            logger.error(f"Error training probability model: {str(e)}", exc_info=True)
            raise

    def optimize_premiums(self, expense_load: float = 0.1, profit_margin: float = 0.15):
        """Calculate risk-adjusted premiums as: Prob(Claim) * E[Severity | Claim] * (1 + expense_load + profit_margin)"""
        if not self.severity_model or not self.probability_model:
            raise ValueError("Train severity and probability models first")

        logger.info("Calculating optimal premiums for all policies")

        try:
            # 1. Get probability predictions for ALL policies
            # Use self.df (full original data) for probability prediction
            X_all_unprocessed, _, full_data_indices = self._get_X_for_prediction(
                self.df, target_col_for_dropping='HasClaim', apply_claim_filter=False)

            # If X_all_unprocessed is empty, it means no data to predict
            if X_all_unprocessed.empty:
                logger.warning("No data found for premium calculation. Returning empty series.")
                return pd.Series([], dtype=float)

            # Use the probability model's preprocessor to transform the full unprocessed X
            prob_preprocessor = self.probability_model.named_steps['preprocessor']
            X_all_processed = prob_preprocessor.transform(X_all_unprocessed)

            # Predict probabilities for the positive class (index 1)
            proba = self.probability_model.named_steps['classifier'].predict_proba(X_all_processed)[:, 1]
            proba_series = pd.Series(proba, index=full_data_indices)  # Attach to original full data index

            # 2. Get severity predictions
            # The severity model was trained only on data where TotalClaims > 0.
            # So, we prepare data *only* for policies with historical claims for severity prediction.
            X_claims_unprocessed, _, claims_data_indices = self._get_X_for_prediction(
                self.df, target_col_for_dropping='TotalClaims', apply_claim_filter=True)

            # Initialize a Series for severity predictions, defaulting to 0 for all policies.
            # This series will be updated with actual severity predictions for policies with claims.
            severity_pred_full_index = pd.Series(0.0, index=self.df.index)

            if not X_claims_unprocessed.empty:
                # Use the severity model's preprocessor to transform the unprocessed X for claims
                sev_preprocessor = self.severity_model.named_steps['preprocessor']
                X_claims_processed = sev_preprocessor.transform(X_claims_unprocessed)

                # Predict raw (log-transformed) severity and then exponentiate back
                severity_preds_raw = self.severity_model.named_steps['regressor'].predict(X_claims_processed)
                severity_preds = np.expm1(severity_preds_raw)

                # Create a Series of severity predictions with their original indices
                severity_preds_series = pd.Series(severity_preds, index=claims_data_indices)

                # Populate the full severity prediction series using the actual predictions for claimed policies
                severity_pred_full_index.loc[claims_data_indices] = severity_preds_series

            # 3. Calculate expected loss: Prob(Claim) * E[Severity | Claim]
            # Both series (proba_series and severity_pred_full_index) are aligned by the original DataFrame index.
            expected_loss = proba_series * severity_pred_full_index

            # Add expense loading and profit margin to get the optimal premium
            optimal_premium = expected_loss * (1 + expense_load + profit_margin)

            return optimal_premium  # Return as Series with original index

        except Exception as e:
            logger.error(f"Error calculating premiums: {str(e)}")
            raise

    def explain_model(self, model_type: str = 'severity'):
        """Generate SHAP explanations for the specified model"""
        # Determine the model, target column, and claim filter setting based on model_type
        if model_type == 'severity':
            model = self.severity_model
            if not model: raise ValueError("Severity model not trained yet")
            target_col = 'TotalClaims'
            claim_filter_val = True
            feature_names = self.severity_feature_names_out  # Use stored feature names
        elif model_type == 'probability':
            model = self.probability_model
            if not model: raise ValueError("Probability model not trained yet")
            target_col = 'HasClaim'
            claim_filter_val = False
            feature_names = self.probability_feature_names_out  # Use stored feature names
        else:
            raise ValueError("Invalid model_type. Must be 'severity' or 'probability'.")

        logger.info(f"Generating SHAP explanations for {model_type} model")

        try:
            # Prepare data (unprocessed X) using the helper, applying claim filter if necessary
            X_unprocessed, cols_for_preprocessor, _ = self._get_X_for_prediction(
                self.df, target_col_for_dropping=target_col, apply_claim_filter=claim_filter_val)

            if X_unprocessed.empty:
                logger.warning(f"No data to explain for {model_type} model after filtering. Skipping SHAP explanation.")
                return

            # Get the fitted preprocessor and the actual predictor from the trained pipeline model
            preprocessor = model.named_steps['preprocessor']
            predictor = model.named_steps['regressor' if model_type == 'severity' else 'classifier']

            # Transform X_unprocessed using the fitted preprocessor to get processed features for SHAP
            X_processed = preprocessor.transform(X_unprocessed)

            # Fallback for feature names in case they weren't stored (though they should be now)
            if feature_names is None or len(feature_names) == 0:
                feature_names = preprocessor.get_feature_names_out(cols_for_preprocessor)
                logger.warning("Feature names were not pre-stored; derived them dynamically for SHAP explanation.")

            # Initialize SHAP explainer based on the predictor type for optimal performance
            if isinstance(predictor, (RandomForestRegressor, RandomForestClassifier, XGBRegressor, XGBClassifier)):
                explainer = shap.TreeExplainer(predictor)
                shap_values = explainer.shap_values(X_processed)
            elif isinstance(predictor, LinearRegression):
                # For LinearRegression, pass X_processed as background data
                explainer = shap.LinearExplainer(predictor, X_processed)
                shap_values = explainer.shap_values(X_processed)
            else:
                # Fallback to KernelExplainer, potentially sampling for performance on large datasets
                logger.warning(
                    "Using KernelExplainer, which can be slow for large datasets. Consider sampling X_processed.")
                # Sample X_processed for KernelExplainer's background data (e.g., 100 samples)
                background_data = shap.sample(X_processed, min(100, X_processed.shape[0]))
                explainer = shap.KernelExplainer(predictor.predict, background_data)
                shap_values = explainer.shap_values(X_processed)

            # Generate and save SHAP summary plot
            plt.figure(figsize=(12, 8))
            # If shap_values is a list (e.g., for multi-class classification), select the positive class's values
            if isinstance(shap_values, list) and len(shap_values) > 1 and model_type == 'probability':
                shap.summary_plot(shap_values[1], X_processed, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)

            plt.title(f'SHAP Summary - {model_type.capitalize()} Model')
            plt.tight_layout()
            plt.savefig(f'shap_{model_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP plot saved as shap_{model_type}.png")

        except Exception as e:
            logger.error(f"Error generating SHAP explanation for {model_type} model: {str(e)}")
            raise
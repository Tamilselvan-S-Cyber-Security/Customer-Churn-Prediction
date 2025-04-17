import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_model(X_train, y_train, model_type, model_params):
    """
    Train a machine learning model for churn prediction

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target variable
    model_type : str
        Type of model to train ('Random Forest', 'Logistic Regression', 'Gradient Boosting', 'XGBoost')
    model_params : dict
        Parameters for the selected model

    Returns:
    --------
    model, feature_importance
        Trained model and feature importance scores
    """
    # Initialize the model based on type
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=42
        )

    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 100),
            random_state=42
        )

    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            learning_rate=model_params.get('learning_rate', 0.1),
            max_depth=model_params.get('max_depth', 3),
            random_state=42
        )

    elif model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            learning_rate=model_params.get('learning_rate', 0.1),
            max_depth=model_params.get('max_depth', 3),
            random_state=42
        )

    # Train the model
    model.fit(X_train, y_train)

    # Extract feature importance
    feature_importance = get_feature_importance(model, X_train.columns)

    return model, feature_importance

def predict_churn(model, X):
    """
    Make churn predictions using the trained model with enhanced error handling
    """
    try:
        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Fill any remaining NaN values with 0
        X = X.fillna(0)

        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.predict(X)

        # Get predicted classes
        y_pred = model.predict(X)

        return y_pred, y_prob

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model

    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    feature_names : list
        List of feature names

    Returns:
    --------
    dict
        Dictionary mapping feature names to importance scores
    """
    # Different models store feature importance differently
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        # If model doesn't provide feature importance, return zeros
        importances = np.zeros(len(feature_names))

    # Create dictionary mapping feature names to importance scores
    feature_importance = dict(zip(feature_names, importances))

    # Sort by importance (descending)
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

    return feature_importance
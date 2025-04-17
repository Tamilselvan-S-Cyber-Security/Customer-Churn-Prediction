import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def get_detailed_feature_explanation(model, feature_names, importance_values, top_n=5):
    """
    Generate detailed explanations for the most important features.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    feature_names : list
        Names of the features
    importance_values : dict
        Dictionary mapping feature names to importance scores
    top_n : int
        Number of top features to explain
    
    Returns:
    --------
    dict
        Dictionary containing feature explanations
    """
    # Sort features by importance
    sorted_features = sorted(importance_values.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Prepare explanations
    explanations = {}
    
    # Feature explanation templates
    feature_explanations = {
        'Gender': "Gender is an important factor in churn prediction. Demographic factors often influence customer behavior and loyalty patterns.",
        'Age': "Age significantly impacts churn likelihood. Different age groups show varying levels of service loyalty and price sensitivity.",
        'Tenure': "Tenure (how long a customer has been with your service) is crucial. Typically, longer-term customers are less likely to churn.",
        'MonthlyCharges': "Monthly charges directly affect customer retention. Higher charges might increase churn risk, especially without corresponding value.",
        'TotalCharges': "Total charges reflect the customer's overall financial investment in your service, which can influence their decision to stay or leave.",
        'Contract': "Contract type is a key predictor. Month-to-month contracts typically show higher churn rates than longer-term contracts.",
        'PaymentMethod': "Payment method can indicate customer commitment level and satisfaction with billing processes.",
        'PaperlessBilling': "Billing preferences might reflect customer engagement with digital services and overall service satisfaction.",
        'OnlineSecurity': "Security features can significantly impact customer retention, especially in digital services.",
        'TechSupport': "Access to technical support affects customer experience and problem resolution, directly impacting churn.",
        'InternetService': "The type of internet service impacts customer experience and perceived value.",
        'OnlineBackup': "Value-added services like online backup can increase the switching cost for customers.",
        'DeviceProtection': "Protection services might indicate higher investment in the relationship and increase retention.",
        'StreamingTV': "Usage patterns for streaming services can reflect how integral your service is to the customer's daily life.",
        'StreamingMovies': "Heavy users of streaming content may have different loyalty patterns compared to light users.",
        'MultipleLines': "Customers with multiple lines may be more invested in your service and face higher switching costs.",
        'PhoneService': "The type of phone service can influence customer satisfaction and perceived value."
    }
    
    # Generic explanation for unknown features
    generic_explanation = "This feature shows significant correlation with customer churn patterns. Its importance suggests it captures meaningful information about customer behavior or preferences that influence their decision to stay or leave."
    
    for feature, importance in top_features:
        feature_key = next((k for k in feature_explanations.keys() if k in feature), None)
        if feature_key:
            explanation = feature_explanations[feature_key]
        else:
            explanation = generic_explanation
            
        impact_level = "high" if importance > 0.2 else "moderate" if importance > 0.1 else "mild"
        explanations[feature] = {
            "importance": importance,
            "explanation": explanation,
            "impact_level": impact_level
        }
    
    return explanations

def analyze_customer_profile(customer_data, feature_importance):
    """
    Analyze a customer profile to identify key characteristics that might influence churn.
    
    Parameters:
    -----------
    customer_data : pandas.DataFrame
        Data for a single customer
    feature_importance : dict
        Dictionary mapping feature names to importance scores
    
    Returns:
    --------
    dict
        Profile analysis with key insights
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:5]]
    
    profile_analysis = {
        "key_features": {},
        "insights": []
    }
    
    # Extract the customer's values for top features
    for feature in top_features:
        if feature in customer_data.columns:
            value = customer_data[feature].iloc[0]
            profile_analysis["key_features"][feature] = value
    
    # Generate insights based on customer profile
    # Contract type insights
    if 'Contract' in customer_data.columns:
        contract = customer_data['Contract'].iloc[0]
        if 'month-to-month' in str(contract).lower():
            profile_analysis["insights"].append("Customer is on a month-to-month contract, which typically has higher churn risk compared to long-term contracts.")
        elif 'one year' in str(contract).lower():
            profile_analysis["insights"].append("Customer is on a one-year contract, providing moderate commitment but will need renewal attention as the term nears completion.")
        elif 'two year' in str(contract).lower():
            profile_analysis["insights"].append("Customer is on a two-year contract, indicating stronger commitment and likely lower short-term churn risk.")
    
    # Tenure insights
    if 'Tenure' in customer_data.columns:
        tenure = customer_data['Tenure'].iloc[0]
        if tenure < 6:
            profile_analysis["insights"].append(f"Customer has only been with the service for {tenure} months. Early-stage customers often have higher churn rates.")
        elif tenure < 24:
            profile_analysis["insights"].append(f"Customer has moderate tenure ({tenure} months), has moved past the high-risk early period.")
        else:
            profile_analysis["insights"].append(f"Customer has high tenure ({tenure} months), indicating established loyalty and potentially lower churn risk.")
    
    # Payment method insights
    if 'PaymentMethod' in customer_data.columns:
        payment = customer_data['PaymentMethod'].iloc[0]
        if 'electronic' in str(payment).lower():
            profile_analysis["insights"].append("Customer uses electronic payment methods, which typically indicate higher digital engagement.")
        elif 'automatic' in str(payment).lower():
            profile_analysis["insights"].append("Customer uses automatic payment methods, which can reduce friction in billing and potentially lower churn risk.")
        elif 'mail' in str(payment).lower():
            profile_analysis["insights"].append("Customer uses mailed checks, which might indicate less digital engagement or preference for traditional methods.")
    
    # Service usage insights
    service_features = [col for col in customer_data.columns if any(service in col for service in 
                                                  ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                                   'TechSupport', 'StreamingTV', 'StreamingMovies'])]
    
    if service_features:
        active_services = []
        for feature in service_features:
            value = customer_data[feature].iloc[0]
            if str(value).lower() == 'yes':
                active_services.append(feature)
        
        if len(active_services) == 0:
            profile_analysis["insights"].append("Customer has minimal add-on services, suggesting lower engagement with the full product ecosystem.")
        elif len(active_services) <= 2:
            profile_analysis["insights"].append(f"Customer uses {len(active_services)} add-on services, indicating moderate engagement with the product ecosystem.")
        else:
            profile_analysis["insights"].append(f"Customer uses {len(active_services)} add-on services, suggesting higher engagement and potentially higher switching costs.")
    
    # Charges insights
    if 'MonthlyCharges' in customer_data.columns:
        monthly_charges = customer_data['MonthlyCharges'].iloc[0]
        if monthly_charges > 80:
            profile_analysis["insights"].append(f"Customer has relatively high monthly charges (${monthly_charges:.2f}), which may increase price sensitivity.")
        elif monthly_charges < 30:
            profile_analysis["insights"].append(f"Customer has lower monthly charges (${monthly_charges:.2f}), potentially indicating a basic service package.")
    
    return profile_analysis

def generate_retention_recommendations(churn_probability, customer_data, model_type):
    """
    Generate personalized retention strategies based on churn probability and customer profile.
    
    Parameters:
    -----------
    churn_probability : float
        Predicted probability of churn
    customer_data : pandas.DataFrame
        Data for a single customer
    model_type : str
        Type of model used for prediction
    
    Returns:
    --------
    dict
        Retention recommendations and risk assessment
    """
    risk_level = "HIGH" if churn_probability > 0.7 else "MEDIUM" if churn_probability > 0.3 else "LOW"
    
    recommendations = {
        "risk_level": risk_level,
        "churn_probability": churn_probability,
        "immediate_actions": [],
        "strategic_recommendations": [],
        "explanation": ""
    }
    
    # Set risk-specific explanation
    if risk_level == "HIGH":
        recommendations["explanation"] = f"This customer has a {churn_probability:.1%} probability of churning, indicating high risk. Immediate intervention is recommended to prevent customer loss."
    elif risk_level == "MEDIUM":
        recommendations["explanation"] = f"This customer has a {churn_probability:.1%} probability of churning, indicating moderate risk. Proactive engagement can help improve retention."
    else:
        recommendations["explanation"] = f"This customer has a {churn_probability:.1%} probability of churning, indicating low risk. Standard engagement practices should be maintained."
    
    # Generate immediate actions based on customer profile
    # Contract-based recommendations
    if 'Contract' in customer_data.columns:
        contract = customer_data['Contract'].iloc[0]
        if 'month-to-month' in str(contract).lower() and risk_level in ["HIGH", "MEDIUM"]:
            recommendations["immediate_actions"].append("Offer incentives to upgrade to a longer-term contract with promotional pricing")
    
    # Tenure-based recommendations
    if 'Tenure' in customer_data.columns:
        tenure = customer_data['Tenure'].iloc[0]
        if tenure < 12 and risk_level == "HIGH":
            recommendations["immediate_actions"].append("Implement an early-stage customer success program with personalized onboarding and check-ins")
        elif 12 <= tenure < 24 and risk_level in ["HIGH", "MEDIUM"]:
            recommendations["immediate_actions"].append("Recognize loyalty milestone and offer tenure-based rewards or upgrades")
    
    # Service-based recommendations
    service_features = [col for col in customer_data.columns if any(service in col for service in 
                                                  ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                                   'TechSupport', 'StreamingTV', 'StreamingMovies'])]
    
    active_services = []
    for feature in service_features:
        if feature in customer_data.columns:
            value = customer_data[feature].iloc[0]
            if str(value).lower() == 'yes':
                active_services.append(feature)
    
    # If few active services and high risk
    if len(active_services) <= 2 and risk_level in ["HIGH", "MEDIUM"]:
        missing_services = [s for s in service_features if s not in active_services]
        if 'OnlineSecurity' in missing_services or 'OnlineBackup' in missing_services:
            recommendations["immediate_actions"].append("Offer a free trial of security or backup services to increase customer value perception")
        
        if 'TechSupport' in missing_services and risk_level == "HIGH":
            recommendations["immediate_actions"].append("Provide one-time complimentary technical support session to address any service issues")
    
    # Charges-based recommendations
    if 'MonthlyCharges' in customer_data.columns:
        monthly_charges = customer_data['MonthlyCharges'].iloc[0]
        if monthly_charges > 80 and risk_level == "HIGH":
            recommendations["immediate_actions"].append("Review current plan for optimization opportunities or targeted discounts")
    
    # Strategic recommendations
    # Add generic strategic recommendations
    strategic_recommendations = [
        "Develop a personalized communication plan based on usage patterns and preferences",
        "Create targeted loyalty rewards aligned with the customer's service usage",
        "Implement regular satisfaction check-ins to address concerns proactively",
        "Analyze usage patterns to identify opportunities for service enhancement",
        "Consider bundled offerings that provide better value for the customer's specific needs"
    ]
    
    # Filter strategic recommendations based on risk level
    if risk_level == "HIGH":
        recommendations["strategic_recommendations"] = strategic_recommendations[:3]
    elif risk_level == "MEDIUM":
        recommendations["strategic_recommendations"] = strategic_recommendations[1:4]
    else:
        recommendations["strategic_recommendations"] = strategic_recommendations[2:4]
    
    return recommendations

def explain_model_decision(model, customer_data, preprocessed_data, feature_names, churn_prediction, churn_probability):
    """
    Explain how the model arrived at a specific churn prediction for a customer.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    customer_data : pandas.DataFrame
        Raw data for a single customer
    preprocessed_data : pandas.DataFrame
        Preprocessed data for the customer
    feature_names : list
        Names of the features after preprocessing
    churn_prediction : int/bool
        The predicted churn outcome (1/True = will churn, 0/False = won't churn)
    churn_probability : float
        The predicted probability of churn
    
    Returns:
    --------
    dict
        Model decision explanation
    """
    explanation = {
        "prediction": "Will Churn" if churn_prediction else "Will Not Churn",
        "confidence": churn_probability if churn_prediction else 1 - churn_probability,
        "confidence_statement": f"The model is {churn_probability*100:.1f}% confident this customer will churn." if churn_prediction else 
                               f"The model is {(1-churn_probability)*100:.1f}% confident this customer will not churn.",
        "key_factors": [],
        "model_strengths": [],
        "model_limitations": []
    }
    
    # Add model type-specific information
    model_type = type(model).__name__
    
    if "RandomForest" in model_type:
        explanation["model_type"] = "Random Forest"
        explanation["model_strengths"] = [
            "Considers complex interactions between customer attributes",
            "Robust to outliers in the data",
            "Combines multiple decision paths to make predictions"
        ]
        explanation["model_limitations"] = [
            "May not capture some linear relationships as efficiently as other models",
            "Can sometimes overfit to patterns in the training data"
        ]
    elif "LogisticRegression" in model_type:
        explanation["model_type"] = "Logistic Regression"
        explanation["model_strengths"] = [
            "Focuses on the strongest linear relationships with churn",
            "Provides straightforward feature importance interpretation",
            "Less prone to overfitting compared to more complex models"
        ]
        explanation["model_limitations"] = [
            "May miss complex non-linear patterns in customer behavior",
            "Assumes features contribute independently to churn probability"
        ]
    elif "GradientBoosting" in model_type:
        explanation["model_type"] = "Gradient Boosting"
        explanation["model_strengths"] = [
            "Sequentially corrects prediction errors",
            "Captures complex relationships in customer data",
            "Often achieves high predictive accuracy"
        ]
        explanation["model_limitations"] = [
            "Can be sensitive to noisy data",
            "May require more data to avoid overfitting"
        ]
    elif "XGB" in model_type:
        explanation["model_type"] = "XGBoost"
        explanation["model_strengths"] = [
            "Advanced implementation of gradient boosting",
            "Handles complex patterns efficiently",
            "Includes regularization to prevent overfitting"
        ]
        explanation["model_limitations"] = [
            "May be sensitive to hyperparameter choices",
            "Complex model behavior can be challenging to interpret fully"
        ]
    else:
        explanation["model_type"] = "Machine Learning Model"
        explanation["model_strengths"] = [
            "Identifies patterns across multiple customer attributes",
            "Makes predictions based on historical churn patterns",
            "Can discover non-obvious relationships in customer data"
        ]
        explanation["model_limitations"] = [
            "Limited to patterns present in the training data",
            "May not capture recent changes in customer behavior"
        ]
    
    # Get key factors affecting the prediction
    # For simplicity, we'll use the feature importance from the model if available
    # A more advanced implementation would use SHAP values or feature permutation
    
    # Simplified approach: use global feature importance for key factors
    if hasattr(model, 'feature_importances_'):
        # Tree-based models have feature_importances_ attribute
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        # Linear models have coef_ attribute
        importances = np.abs(model.coef_[0])
        feature_importance = dict(zip(feature_names, importances))
    else:
        # Fallback to permutation importance (simplified version)
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = 1.0 / (i + 1)  # Dummy values
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:5]
    
    # Translate feature names back to original names when possible
    # In a real implementation, you would maintain a mapping between original and transformed features
    for feature, importance in top_features:
        # Extract the original feature name from the transformed feature if possible
        original_feature = feature
        for col in customer_data.columns:
            if col in feature:
                original_feature = col
                break
        
        # Get the customer's value for this feature
        customer_value = None
        if original_feature in customer_data.columns:
            customer_value = customer_data[original_feature].iloc[0]
        
        factor = {
            "feature": original_feature,
            "importance": importance,
            "customer_value": customer_value,
            "contribution": "Increases churn risk" if importance > 0.2 else "Moderate impact on churn" if importance > 0.1 else "Minor impact on churn"
        }
        
        explanation["key_factors"].append(factor)
    
    return explanation

def generate_model_overview(model, metrics, feature_importance):
    """
    Generate an overview of the churn prediction model and its performance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    metrics : dict
        Model performance metrics
    feature_importance : dict
        Feature importance scores
    
    Returns:
    --------
    dict
        Model overview with performance analysis
    """
    model_type = type(model).__name__
    
    if "RandomForest" in model_type:
        model_name = "Random Forest"
        approach = "Ensemble of decision trees that vote on the final prediction"
    elif "LogisticRegression" in model_type:
        model_name = "Logistic Regression"
        approach = "Linear model that predicts churn probability based on feature weights"
    elif "GradientBoosting" in model_type:
        model_name = "Gradient Boosting"
        approach = "Sequential ensemble that builds trees to correct previous errors"
    elif "XGB" in model_type:
        model_name = "XGBoost"
        approach = "Advanced gradient boosting implementation with regularization"
    else:
        model_name = "Machine Learning Model"
        approach = "Algorithm that identifies patterns in customer data to predict churn"
    
    # Interpret model performance
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    
    if accuracy > 0.9:
        accuracy_assessment = "excellent"
    elif accuracy > 0.8:
        accuracy_assessment = "good"
    elif accuracy > 0.7:
        accuracy_assessment = "moderate"
    else:
        accuracy_assessment = "fair"
    
    if precision > 0.9:
        precision_assessment = "exceptional"
    elif precision > 0.8:
        precision_assessment = "very good"
    elif precision > 0.7:
        precision_assessment = "good"
    else:
        precision_assessment = "moderate"
    
    if recall > 0.9:
        recall_assessment = "exceptional"
    elif recall > 0.8:
        recall_assessment = "very good"
    elif recall > 0.7:
        recall_assessment = "good"
    else:
        recall_assessment = "moderate"
    
    # Get top features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    overview = {
        "model_name": model_name,
        "approach": approach,
        "performance": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy_assessment": accuracy_assessment,
            "precision_assessment": precision_assessment,
            "recall_assessment": recall_assessment
        },
        "top_features": [{"name": name, "importance": importance} for name, importance in top_features],
        "summary": (
            f"This {model_name} model predicts customer churn with {accuracy_assessment} accuracy ({accuracy:.1%}). "
            f"When it predicts a customer will churn, it is correct {precision:.1%} of the time ({precision_assessment} precision). "
            f"It successfully identifies {recall:.1%} of all customers who actually churn ({recall_assessment} recall)."
        ),
        "business_insight": (
            f"The model has identified {top_features[0][0]}, {top_features[1][0]}, and {top_features[2][0]} "
            f"as the most important factors influencing customer churn. "
            f"Focus retention strategies on these areas for maximum impact."
        )
    }
    
    return overview
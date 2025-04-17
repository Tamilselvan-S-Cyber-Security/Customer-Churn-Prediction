import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_feature_importance(feature_importance, feature_names):
    """
    Create a bar chart of feature importance
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary of feature importance scores
    feature_names : list
        List of feature names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Feature importance plot
    """
    # Convert dictionary to dataframe for plotting
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Limit to top 15 features for readability
    if len(df) > 15:
        df = df.head(15)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout for dark theme
    fig.update_layout(
        plot_bgcolor='#262730',
        paper_bgcolor='#0E1117',
        font_color='white',
        coloraxis_colorbar_thickness=15,
        margin=dict(l=10, r=10, t=50, b=10),
        height=500
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix visualization
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        hoverongaps=False,
        colorscale='Viridis',
        showscale=False,
        text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
        texttemplate="%{text}"
    ))
    
    # Update layout
    fig.update_layout(
        title='Confusion Matrix',
        plot_bgcolor='#262730',
        paper_bgcolor='#0E1117',
        font_color='white',
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        xaxis=dict(title='Predicted Values'),
        yaxis=dict(title='Actual Values')
    )
    
    return fig

def plot_roc_curve(y_true, y_score):
    """
    Create a ROC curve visualization
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_score : array-like
        Target scores (probability estimates)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        ROC curve plot
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#FF4B4B', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        plot_bgcolor='#262730',
        paper_bgcolor='#0E1117',
        font_color='white',
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
        shapes=[
            dict(
                type='line',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='gray', width=1, dash='dash')
            )
        ]
    )
    
    return fig

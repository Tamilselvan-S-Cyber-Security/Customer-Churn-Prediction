import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import io
import base64
import os

# Import custom modules
from utils.preprocessing import preprocess_data, split_data, handle_missing_values, encode_categorical_features
from utils.model import train_model, predict_churn, get_feature_importance
from utils.visualization import plot_feature_importance, plot_confusion_matrix, plot_roc_curve
from utils.explanations import (get_detailed_feature_explanation, analyze_customer_profile, 
                              generate_retention_recommendations, explain_model_decision,
                              generate_model_overview)

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("assets/user-check.svg", width=100)
    st.title("Customer Churn Prediction")
    st.subheader("Upload your data and predict customer churn")

    # Data source selection
    data_source = st.radio(
        "Choose data source",
        ["Use sample dataset", "Upload your own CSV"],
        index=0,
        key="data_source_selection"
    )

    if data_source == "Use sample dataset":
        try:
            df = pd.read_csv("assets/sample_churn_data.csv")
            st.success("✅ Sample churn data loaded successfully!")
            uploaded_file = "assets/sample_churn_data.csv"
            st.session_state.data = df

            try:
                # Show sample data download link
                sample_data_path = "assets/sample_churn_data.csv"
                if os.path.exists(sample_data_path):
                    with open(sample_data_path, "r") as f:
                        csv_data = f.read()
                        st.download_button(
                            label="Download sample data (for reference)",
                            data=csv_data,
                            file_name="sample_churn_data.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Sample data file not found.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
                uploaded_file = None
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            uploaded_file = None
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your CSV file", 
                type=['csv'],
                key="csv_uploader",
                help="Upload a CSV file with customer data",
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                try:
                    # Read the file content into memory
                    file_content = uploaded_file.read()
                    # Convert to StringIO for pandas
                    import io
                    string_data = io.StringIO(file_content.decode('utf-8'))
                    try:
                        df = pd.read_csv(string_data)
                    except UnicodeDecodeError:
                        # Try with different encoding if UTF-8 fails
                        string_data = io.StringIO(file_content.decode('latin1'))
                        df = pd.read_csv(string_data)

                    # Basic validation
                    if len(df.columns) < 2:
                        st.error("❌ CSV file must contain at least 2 columns")
                        uploaded_file = None
                    else:
                        st.success("✅ File uploaded successfully!")
                        st.session_state.data = df
                        with col2:
                            st.metric("Rows", len(df))
                            st.metric("Columns", len(df.columns))
                except Exception as e:
                    st.error("❌ Error reading the file. Please ensure it's a valid CSV.")
                    st.info("💡 Tip: Download and check the sample data format above.")
                    uploaded_file = None

        if uploaded_file is not None:
            try:
                # Read CSV file with more robust error handling
                df = pd.read_csv(uploaded_file, na_values=['NA', 'missing', '', ' '], low_memory=False)

                # Basic data validation
                if len(df.columns) < 2:
                    st.error("❌ CSV file must contain at least 2 columns")
                    uploaded_file = None
                else:
                    st.success("✅ File uploaded successfully!")

                    # Show data preview
                    st.write("Data Preview:")
                    st.dataframe(df.head())

                    # Display basic statistics
                    st.write("Basic Statistics:")
                    st.write({
                        "Total Records": len(df),
                        "Total Features": len(df.columns),
                        "Missing Values": df.isnull().sum().sum()
                    })

                    # Store in session state
                    st.session_state.data = df

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format")
                uploaded_file = None


    # Model selection
    model_type = st.selectbox(
        "Select ML Model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost"]
    )

    # Model parameters
    st.subheader("Model Parameters")
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of trees", 50, 300, 100, 10)
        max_depth = st.slider("Maximum depth", 2, 20, 10, 1)
        model_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    elif model_type == "Logistic Regression":
        C = st.slider("Regularization strength", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.slider("Maximum iterations", 100, 1000, 100, 50)
        model_params = {"C": C, "max_iter": max_iter}

    elif model_type == "Gradient Boosting":
        n_estimators = st.slider("Number of boosting stages", 50, 300, 100, 10)
        learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        max_depth = st.slider("Maximum depth", 2, 10, 3, 1)
        model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}

    elif model_type == "XGBoost":
        n_estimators = st.slider("Number of boosting stages", 50, 300, 100, 10)
        learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        max_depth = st.slider("Maximum depth", 2, 10, 3, 1)
        model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}

    # Testing set size
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100

    # Sidebar footer
    st.markdown("---")
    st.markdown("### Developed By")
    st.markdown("**S.Tamilselvan**")
    st.markdown("*Customer Churn Analysis Expert*")

# Main area
st.title("🔍 Customer Churn Prediction")
st.markdown("### Uncover hidden patterns and predict customer attrition")

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Overview", "Preprocessing", "Model Training", "Predictions", "Future Prediction", "AI Explanations"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'y_prob' not in st.session_state:
    st.session_state.y_prob = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Process uploaded file
if uploaded_file is not None:
    try:
        # Check if we're using the sample data or an uploaded file
        if isinstance(uploaded_file, str) and uploaded_file == "assets/sample_churn_data.csv":
            # Load sample data from assets folder
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
        else:
            # Process uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Tab 1: Data Overview
with tab1:
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
            st.metric("Features", len(st.session_state.data.columns))

        with col2:
            st.metric("Missing Values", st.session_state.data.isna().sum().sum())

            # Detect if data already has a churn column
            potential_target_cols = [col for col in st.session_state.data.columns if 'churn' in col.lower() or 'exit' in col.lower() or 'left' in col.lower() or 'attrition' in col.lower()]

            if potential_target_cols:
                default_target = potential_target_cols[0]
            else:
                default_target = st.session_state.data.columns[-1]

            st.session_state.target_column = st.selectbox(
                "Select target column (churn indicator)",
                options=st.session_state.data.columns,
                index=list(st.session_state.data.columns).index(default_target) if default_target in st.session_state.data.columns else 0
            )

        st.subheader("Data Information")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.data.describe())

        # Data visualization section
        st.subheader("Data Visualization")

        # Choose visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            ["Distribution of Target", "Correlation Heatmap", "Feature Histograms"]
        )

        if viz_type == "Distribution of Target":
            if st.session_state.target_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                st.session_state.data[st.session_state.target_column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, textprops={'color': 'white'})
                ax.set_title(f'Distribution of {st.session_state.target_column}', color='white')
                ax.set_ylabel('')
                fig.set_facecolor('#0E1117')
                plt.tight_layout()
                st.pyplot(fig)

        elif viz_type == "Correlation Heatmap":
            # Filter only numeric columns
            numeric_data = st.session_state.data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.title('Correlation Heatmap', color='white')
                mask = np.triu(np.ones_like(numeric_data.corr()))
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax, mask=mask)
                fig.set_facecolor('#0E1117')
                for text in ax.texts:
                    text.set_color('black')
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for correlation analysis.")

        elif viz_type == "Feature Histograms":
            # Allow selecting a column
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for histogram", numeric_cols)
                fig = px.histogram(st.session_state.data, x=selected_col, color=st.session_state.target_column,
                                 title=f"Distribution of {selected_col} by {st.session_state.target_column}")
                fig.update_layout(
                    plot_bgcolor='#262730',
                    paper_bgcolor='#0E1117',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for histogram visualization.")
    else:
        st.info("Please upload a CSV file to view data overview.")

        # Show example data structure
        st.markdown("""
        ### Expected Data Format
        Your CSV file should include customer attributes and a churn indicator column. Example fields:
        - Customer demographics (age, gender, etc.)
        - Account information (tenure, contract type, etc.)
        - Usage patterns
        - Payment history
        - Churn indicator (target column)
        """)

# Tab 2: Preprocessing
with tab2:
    if st.session_state.data is not None:
        st.subheader("Data Preprocessing")

        # Preprocessing options
        st.markdown("### Select Preprocessing Steps")

        col1, col2 = st.columns(2)

        with col1:
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            missing_strategy = st.selectbox(
                "Missing Values Strategy",
                ["Mean/Mode", "Median/Mode", "Drop Rows", "Fill with Zero"]
            )

            encode_categorical = st.checkbox("Encode Categorical Features", value=True)
            encoding_strategy = st.selectbox(
                "Encoding Strategy",
                ["One-Hot Encoding", "Label Encoding"]
            )

        with col2:
            feature_scaling = st.checkbox("Apply Feature Scaling", value=True)
            scaling_method = st.selectbox(
                "Scaling Method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )

            remove_outliers = st.checkbox("Remove Outliers", value=False)
            if remove_outliers:
                outlier_threshold = st.slider("Outlier Threshold (z-score)", 2.0, 5.0, 3.0, 0.1)
            else:
                outlier_threshold = 3.0

        # Feature selection
        st.subheader("Feature Selection")

        # Select features to exclude
        features_to_exclude = st.multiselect(
            "Select features to exclude",
            [col for col in st.session_state.data.columns if col != st.session_state.target_column],
            default=[]
        )

        # Preprocess button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Preprocess data based on selected options
                preprocessing_options = {
                    "handle_missing": handle_missing,
                    "missing_strategy": missing_strategy,
                    "encode_categorical": encode_categorical,
                    "encoding_strategy": encoding_strategy,
                    "feature_scaling": feature_scaling,
                    "scaling_method": scaling_method,
                    "remove_outliers": remove_outliers,
                    "outlier_threshold": outlier_threshold,
                    "features_to_exclude": features_to_exclude,
                    "target_column": st.session_state.target_column
                }

                try:
                    st.session_state.preprocessed_data, feature_names = preprocess_data(
                        st.session_state.data, 
                        preprocessing_options
                    )

                    # Split the data
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = split_data(
                        st.session_state.preprocessed_data,
                        st.session_state.target_column,
                        test_size
                    )

                    st.success("Data preprocessing completed successfully!")

                    # Display preprocessed data preview
                    st.subheader("Preprocessed Data Preview")
                    st.dataframe(st.session_state.preprocessed_data.head(10))

                    # Display shapes
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training Samples", len(st.session_state.X_train))
                    with col2:
                        st.metric("Testing Samples", len(st.session_state.X_test))
                    with col3:
                        st.metric("Features", st.session_state.X_train.shape[1])
                    with col4:
                        churn_rate = round(st.session_state.preprocessed_data[st.session_state.target_column].mean() * 100, 2)
                        st.metric("Churn Rate", f"{churn_rate}%")

                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    else:
        st.info("Please upload a CSV file in the Data Overview tab first.")

# Tab 3: Model Training
with tab3:
    if st.session_state.X_train is not None and st.session_state.y_train is not None:
        st.subheader("Model Training")

        st.markdown(f"**Selected Model:** {model_type}")

        # Display model parameters
        st.json(model_params)

        # Train model button
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type} model..."):
                try:
                    # Train the model
                    model, feature_importance = train_model(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        model_type,
                        model_params
                    )

                    # Save model and feature importance to session state
                    st.session_state.model = model
                    st.session_state.feature_importance = feature_importance

                    # Make predictions on test set
                    y_pred, y_prob = predict_churn(
                        model,
                        st.session_state.X_test
                    )

                    st.session_state.y_pred = y_pred
                    st.session_state.y_prob = y_prob

                    # Calculate metrics
                    accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    precision = precision_score(st.session_state.y_test, y_pred, zero_division=0)
                    recall = recall_score(st.session_state.y_test, y_pred, zero_division=0)
                    f1 = f1_score(st.session_state.y_test, y_pred, zero_division=0)

                    st.session_state.metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                    }

                    st.success("Model training completed successfully!")

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.4f}")

                    # Feature importance visualization
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(feature_importance, st.session_state.X_train.columns)
                    st.plotly_chart(fig, use_container_width=True)

                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm_fig = plot_confusion_matrix(st.session_state.y_test, y_pred)
                    st.plotly_chart(cm_fig, use_container_width=True)

                    # ROC curve
                    st.subheader("ROC Curve")
                    roc_fig = plot_roc_curve(st.session_state.y_test, y_prob)
                    st.plotly_chart(roc_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during model training: {e}")

        # Display trained model results if available
        if st.session_state.metrics is not None:
            st.subheader("Model Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{st.session_state.metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{st.session_state.metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{st.session_state.metrics['f1']:.4f}")

            if st.session_state.feature_importance is not None:
                st.subheader("Feature Importance")
                fig = plot_feature_importance(st.session_state.feature_importance, st.session_state.X_train.columns)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please complete data preprocessing in the Preprocessing tab first.")

# Tab 4: Predictions
with tab4:
    if st.session_state.model is not None and st.session_state.y_pred is not None:
        st.subheader("Model Predictions")

        # Display prediction results
        prediction_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.y_pred,
            'Probability': st.session_state.y_prob
        })

        st.dataframe(prediction_df)

        # Visualization options
        viz_option = st.selectbox(
            "Select visualization",
            ["Confusion Matrix", "ROC Curve", "Prediction Distribution"]
        )

        if viz_option == "Confusion Matrix":
            cm_fig = plot_confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            st.plotly_chart(cm_fig, use_container_width=True)

        elif viz_option == "ROC Curve":
            roc_fig = plot_roc_curve(st.session_state.y_test, st.session_state.y_prob)
            st.plotly_chart(roc_fig, use_container_width=True)

        elif viz_option == "Prediction Distribution":
            fig = px.histogram(prediction_df, x="Probability", color="Actual", 
                               marginal="box", 
                               title="Distribution of Prediction Probabilities")
            fig.update_layout(
                plot_bgcolor='#262730',
                paper_bgcolor='#0E1117',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Export results
        st.subheader("Export Prediction Results")

        csv = prediction_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv" class="btn">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.info("Please train a model in the Model Training tab first.")

# Tab 5: Future Prediction
with tab5:
    st.subheader("Future Customer Churn Prediction")

    if st.session_state.model is not None:
        # Create tabs for different prediction methods
        pred_tab1, pred_tab2 = st.tabs(["CSV Upload", "Manual Input Form"])

        # Tab 1: CSV Upload for predictions
        with pred_tab1:
            st.markdown("""
            ### Upload New Customer Data
            Upload a CSV file with new customer data to predict churn. 
            The file should have the same structure as your training data, but without the target column.
            """)

            # Use sample data by default - this is the most reliable method
            use_sample_data_for_prediction = st.checkbox("Use sample churn dataset for prediction", value=True)

            if use_sample_data_for_prediction:
                new_data_file = "assets/sample_churn_data.csv"
                st.success("✅ Sample data loaded for prediction!")

                # Show a download link for the sample data
                try:
                    sample_data_path = "assets/sample_churn_data.csv"
                    if os.path.exists(sample_data_path):
                        with open(sample_data_path, "r") as f:
                            csv_data = f.read()
                            st.download_button(
                                label="Download sample data (for reference)",
                                data=csv_data,
                                file_name="sample_churn_data.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning("Sample data file not found.")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

                st.info("If you need to use your own data, please download the sample file, modify it according to your needs, and then upload your modified file when running locally.")
            else:
                st.error("⚠️ Please use the sample data option above. Direct file uploads are currently unavailable due to system restrictions.")
                new_data_file = None

            if new_data_file is not None:
                try:
                    # Check if we're using the sample data or an uploaded file
                    if isinstance(new_data_file, str) and new_data_file == "assets/sample_churn_data.csv":
                        # Load sample data from assets folder
                        new_data = pd.read_csv(new_data_file)
                    else:
                        # Process uploaded file
                        new_data = pd.read_csv(new_data_file)

                    # Show new data preview
                    st.subheader("New Data Preview")
                    st.dataframe(new_data.head(10))

                    # Check if target column exists in new data and remove it
                    if st.session_state.target_column in new_data.columns:
                        st.warning(f"Target column '{st.session_state.target_column}' found in new data and will be ignored for prediction.")
                        new_data = new_data.drop(columns=[st.session_state.target_column])

                    # Preprocess new data
                    if st.button("Predict Churn for New Data"):
                        with st.spinner("Processing and predicting..."):
                            try:
                                # Preprocess the new data (assuming you've saved preprocessing steps)
                                # For simplicity, we'll reuse the preprocessing function
                                preprocessing_options = {
                                    "handle_missing": True,
                                    "missing_strategy": "Mean/Mode",
                                    "encode_categorical": True,
                                    "encoding_strategy": "One-Hot Encoding",
                                    "feature_scaling": True,
                                    "scaling_method": "StandardScaler",
                                    "remove_outliers": False,
                                    "outlier_threshold": 3.0,
                                    "features_to_exclude": [],
                                    "target_column": "temp_target"  # Dummy target, will be removed
                                }

                                # Add a temporary target column (will be removed in processing)
                                new_data["temp_target"] = 0

                                # Preprocess new data
                                processed_new_data, _ = preprocess_data(new_data, preprocessing_options)

                                # Remove the temporary target column
                                processed_new_data = processed_new_data.drop(columns=["temp_target"])

                                # Ensure the columns match the training data
                                missing_cols = set(st.session_state.X_train.columns) - set(processed_new_data.columns)
                                for col in missing_cols:
                                    processed_new_data[col] = 0

                                # Keep only the columns that were in the training data
                                processed_new_data = processed_new_data[st.session_state.X_train.columns]

                                # Make predictions
                                new_pred, new_prob = predict_churn(st.session_state.model, processed_new_data)

                                # Add predictions to original data
                                result_df = new_data.copy()
                                result_df["Churn_Prediction"] = new_pred
                                result_df["Churn_Probability"] = new_prob

                                # Display results
                                st.subheader("Prediction Results")
                                st.dataframe(result_df)

                                # Visualization of predictions
                                fig = px.histogram(result_df, x="Churn_Probability", 
                                                   title="Distribution of Churn Probabilities for New Data")
                                fig.update_layout(
                                    plot_bgcolor='#262730',
                                    paper_bgcolor='#0E1117',
                                    font_color='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Show count of predicted churners
                                churn_count = result_df["Churn_Prediction"].sum()
                                total_count = len(result_df)
                                churn_percent = (churn_count / total_count) * 100

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Customers", total_count)
                                with col2:
                                    st.metric("Predicted Churners", churn_count)
                                with col3:
                                    st.metric("Churn Rate", f"{churn_percent:.2f}%")

                                # Export results
                                st.subheader("Export Prediction Results")

                                csv = result_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="new_churn_predictions.csv" class="btn">Download CSV File</a>'
                                st.markdown(href, unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                except Exception as e:
                    st.error(f"Error reading the file: {e}")

        # Tab 2: Manual Input Form
        with pred_tab2:
            st.markdown("### Enter Customer Details Manually")

            if st.session_state.data is not None:
                # Get relevant columns (excluding target)
                input_columns = [col for col in st.session_state.data.columns if col != st.session_state.target_column]

                # Create form for manual input
                with st.form("manual_prediction_form"):
                    st.subheader("Enter Customer Information")

                    # Create input fields dynamically based on data columns
                    input_values = {}

                    # Create two columns for better layout
                    col1, col2 = st.columns(2)

                    # Divide columns between the two visual columns
                    half_point = len(input_columns) // 2

                    for i, col in enumerate(input_columns):
                        # Check column type to determine input widget
                        col_type = st.session_state.data[col].dtype

                                                # Use appropriate column based on index
                        display_col = col1 if i < half_point else col2

                        with display_col:
                            if col_type == 'object' or col_type == 'category':
                                # For categorical columns, create a selectbox with unique values
                                unique_values = st.session_state.data[col].unique().tolist()
                                input_values[col] = st.selectbox(f"{col}", unique_values, key=f"manual_{col}")
                            elif col_type== 'bool':
                                # For boolean columns
                                input_values[col] = st.checkbox(f"{col}", key=f"manual_{col}")
                            elif np.issubdtype(col_type, np.integer):
                                # For integer columns
                                min_val = int(st.session_state.data[col].min())
                                max_val = int(st.session_state.data[col].max())
                                default = int((min_val + max_val) / 2)
                                input_values[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default, step=1, key=f"manual_{col}")
                            else:
                                # For float columns
                                min_val = float(st.session_state.data[col].min())
                                max_val = float(st.session_state.data[col].max())
                                default = (min_val + max_val) / 2
                                input_values[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default, key=f"manual_{col}")

                    # Submit button
                    submitted = st.form_submit_button("Predict Churn")

                    if submitted:
                        try:
                            # Create DataFrame with manual input
                            input_df = pd.DataFrame([input_values])

                            # Add a temporary target column (will be removed in processing)
                            input_df["temp_target"] = 0

                            # Preprocess the input
                            preprocessing_options = {
                                "handle_missing": True,
                                "missing_strategy": "Mean/Mode",
                                "encode_categorical": True,
                                "encoding_strategy": "One-Hot Encoding",
                                "feature_scaling": True,
                                "scaling_method": "StandardScaler",
                                "remove_outliers": False,
                                "outlier_threshold": 3.0,
                                "features_to_exclude": [],
                                "target_column": "temp_target"  # Dummy target
                            }

                            # Preprocess new data
                            processed_input, _ = preprocess_data(input_df, preprocessing_options)

                            # Remove the temporary target column
                            processed_input = processed_input.drop(columns=["temp_target"])

                            # Ensure the columns match the training data
                            missing_cols = set(st.session_state.X_train.columns) - set(processed_input.columns)
                            for col in missing_cols:
                                processed_input[col] = 0

                            # Keep only the columns that were in the training data
                            processed_input = processed_input[st.session_state.X_train.columns]

                            # Make prediction
                            _, prob = predict_churn(st.session_state.model, processed_input)

                            # Display result
                            st.subheader("Churn Prediction Result")

                            # Create gauge chart for churn probability
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = prob[0] * 100,
                                title = {'text': "Churn Probability"},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "red"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "green"},
                                        {'range': [30, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "red"}
                                    ]
                                }
                            ))

                            fig.update_layout(
                                height=300,
                                plot_bgcolor='#0E1117',
                                paper_bgcolor='#0E1117',
                                font_color='white'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Display recommendation based on probability
                            st.subheader("Recommendation")

                            if prob[0] < 0.3:
                                st.success("This customer has a low risk of churning.")
                                st.markdown("**Suggested Actions:**")
                                st.markdown("- Continue standard engagement")
                                st.markdown("- Offer loyalty rewards")
                                st.markdown("- Regular satisfaction surveys")
                            elif prob[0] < 0.7:
                                st.warning("This customer has a moderate risk of churning.")
                                st.markdown("**Suggested Actions:**")
                                st.markdown("- Proactive engagement")
                                st.markdown("- Special offers or discounts")
                                st.markdown("- Service quality check")
                                st.markdown("- Personalized communication")
                            else:
                                st.error("This customer has a high risk of churning.")
                                st.markdown("**Suggested Actions:**")
                                st.markdown("- Immediate intervention")
                                st.markdown("- Retention specialist contact")
                                st.markdown("- Custom retention package")
                                st.markdown("- Service upgrade offers")
                                st.markdown("- Personalized win-back strategy")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
            else:
                st.info("Please upload or use sample data first to enable the manual input form.")
    else:
        st.info("Please train a model in the Model Training tab first.")

# Tab 6: AI Explanations
with tab6:
    st.subheader("AI-Powered Churn Analysis & Explanations")

    if st.session_state.model is not None and st.session_state.data is not None:
        # Introduction
        st.markdown("""
        ### AI-Powered Insights
        This tab provides intelligent explanations and insights about your churn prediction model, 
        helping you understand the factors driving customer churn and develop targeted retention strategies.
        """)

        # Create tabs for different types of explanations
        explain_tab1, explain_tab2, explain_tab3 = st.tabs([
            "Model Explanations", 
            "Customer Profile Analysis", 
            "Retention Recommendations"
        ])

        # Tab 1: Model Explanations
        with explain_tab1:
            st.markdown("### Understanding Your Churn Prediction Model")

            if st.session_state.metrics is not None and st.session_state.feature_importance is not None:
                # Generate model overview and insights
                model_overview = generate_model_overview(
                    st.session_state.model,
                    st.session_state.metrics,
                    st.session_state.feature_importance
                )

                # Display model summary
                st.markdown(f"#### {model_overview['model_name']} Model Summary")
                st.markdown(model_overview['summary'])

                # Display model performance metrics
                st.markdown("#### Model Performance")

                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{model_overview['performance']['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{model_overview['performance']['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{model_overview['performance']['recall']:.2%}")
                with col4:
                    st.metric("F1 Score", f"{model_overview['performance']['f1']:.2%}")

                # Feature importance analysis
                st.markdown("#### Key Churn Factors")
                st.markdown(model_overview['business_insight'])

                # Generate detailed explanations of important features
                feature_explanations = get_detailed_feature_explanation(
                    st.session_state.model,
                    st.session_state.X_train.columns,
                    st.session_state.feature_importance,
                    top_n=5
                )

                # Display feature explanations
                st.markdown("#### Feature Importance Explanations")

                for i, (feature, details) in enumerate(feature_explanations.items()):
                    with st.expander(f"{i+1}. {feature} (Importance: {details['importance']:.4f})"):
                        st.markdown(f"**Impact Level:** {details['impact_level'].title()}")
                        st.markdown(details['explanation'])

                # Model strengths and limitations
                st.markdown("#### Model Strengths & Limitations")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strengths**")
                    for strength in model_overview.get('model_strengths', ['Advanced pattern recognition']):
                        st.markdown(f"- {strength}")

                with col2:
                    st.markdown("**Limitations**")
                    for limitation in model_overview.get('model_limitations', ['Limited to patterns in historical data']):
                        st.markdown(f"- {limitation}")

                # How to use the model effectively
                st.markdown("#### How to Use This Model Effectively")
                st.markdown("""
                1. **Focus on top features**: Concentrate retention efforts on the highest importance factors
                2. **Target high probability churners**: Prioritize customers with the highest churn probability
                3. **Monitor performance**: Regularly check model metrics as customer behavior evolves
                4. **Balance precision and recall**: Adjust intervention thresholds based on your business priorities
                5. **Combine with domain knowledge**: Use model insights alongside your business expertise
                """)

            else:
                st.info("Please train a model first to access AI-powered model explanations.")

        # Tab 2: Customer Profile Analysis
        with explain_tab2:
            st.markdown("### Individual Customer Analysis")

            if st.session_state.model is not None and st.session_state.feature_importance is not None:
                # Option to select a customer from the dataset
                st.markdown("#### Select a customer to analyze")

                # Allow selecting a customer
                if 'CustomerID' in st.session_state.data.columns:
                    customer_id = st.selectbox(
                        "Select Customer ID", 
                        options=st.session_state.data['CustomerID'].unique()
                    )

                    # Get the customer data
                    customer_data = st.session_state.data[st.session_state.data['CustomerID'] == customer_id]
                else:
                    # If no CustomerID, use row index
                    customer_idx = st.selectbox(
                        "Select Customer (by row index)", 
                        options=range(len(st.session_state.data))
                    )

                    # Get the customer data
                    customer_data = st.session_state.data.iloc[[customer_idx]]

                if not customer_data.empty:
                    # Display basic customer information
                    st.markdown("#### Customer Profile")

                    # Extract key demographic info if available
                    profile_cols = [col for col in customer_data.columns if col in 
                                  ['CustomerID', 'Gender', 'Age', 'Tenure', 'Contract', 
                                   'MonthlyCharges', 'TotalCharges']]

                    if profile_cols:
                        st.dataframe(customer_data[profile_cols])
                    else:
                        st.dataframe(customer_data.iloc[:, :5])  # Show first 5 columns

                    # Analyze customer profile
                    profile_analysis = analyze_customer_profile(
                        customer_data, 
                        st.session_state.feature_importance
                    )

                    # Display profile insights
                    st.markdown("#### Key Insights")
                    for insight in profile_analysis['insights']:
                        st.markdown(f"- {insight}")

                    # Make prediction for this customer
                    # Preprocess the customer data (simplified - in real app would need more complex preprocessing)
                    if st.session_state.target_column in customer_data.columns:
                        actual_churn = customer_data[st.session_state.target_column].iloc[0]
                        st.markdown(f"**Actual Churn Status:** {'Yes' if actual_churn else 'No'}")

                    # Add a button to generate a detailed analysis
                    if st.button("Generate Detailed Analysis"):
                        with st.spinner("Analyzing customer profile..."):
                            # In a real implementation, this would use the Anthropic API for more detailed analysis
                            # For now, we'll use our rule-based insights

                            st.markdown("#### Churn Risk Assessment")

                            # For demonstration, we'll create a placeholder risk assessment
                            # In a real app, this would use the model to make an actual prediction
                            if 'Tenure' in customer_data.columns and 'Contract' in customer_data.columns:
                                tenure = customer_data['Tenure'].iloc[0]
                                contract = str(customer_data['Contract'].iloc[0]).lower()

                                if tenure < 12 and 'month' in contract:
                                    risk_level = "High"
                                    risk_color = "🔴"
                                elif tenure < 24 and 'month' in contract:
                                    risk_level = "Medium"
                                    risk_color = "🟠"
                                else:
                                    risk_level = "Low" 
                                    risk_color = "🟢"

                                st.markdown(f"**Churn Risk Level:** {risk_color} {risk_level}")

                            # Generate personalized recommendation
                            st.markdown("#### Personalized Engagement Strategy")
                            st.markdown("""
                            1. **Targeted Communication**: Engage based on usage patterns and preferences
                            2. **Value Demonstration**: Highlight specific benefits most relevant to this customer
                            3. **Proactive Support**: Address potential issues before they trigger churn
                            4. **Customized Offers**: Develop offers aligned with the customer's specific needs
                            """)

            else:
                st.info("Please train a model first to access customer profile analysis.")

        # Tab 3: Retention Recommendations
        with explain_tab3:
            st.markdown("### Smart Retention Strategies")

            if st.session_state.model is not None:
                # Customer selection for retention recommendations
                st.markdown("#### Generate Personalized Retention Strategies")

                # Allow selecting a customer
                if 'CustomerID' in st.session_state.data.columns:
                    retention_customer_id = st.selectbox(
                        "Select Customer ID", 
                        options=st.session_state.data['CustomerID'].unique(),
                        key="retention_customer_id"
                    )

                    # Get the customer data
                    retention_customer_data = st.session_state.data[st.session_state.data['CustomerID'] == retention_customer_id]
                else:
                    # If no CustomerID, use row index
                    retention_customer_idx = st.selectbox(
                        "Select Customer (by row index)", 
                        options=range(len(st.session_state.data)),
                        key="retention_customer_idx"
                    )

                    # Get the customer data
                    retention_customer_data = st.session_state.data.iloc[[retention_customer_idx]]

                if not retention_customer_data.empty:
                    # Generate a simulated churn probability (in a real app, would use model prediction)
                    # For simplicity, we'll generate a random probability based on customer attributes
                    churn_prob = 0.5  # Default

                    if 'Tenure' in retention_customer_data.columns:
                        tenure = retention_customer_data['Tenure'].iloc[0]
                        # Lower tenure generally means higher churn risk
                        if tenure < 12:
                            churn_prob += 0.3
                        elif tenure > 36:
                            churn_prob -= 0.3

                    if 'Contract' in retention_customer_data.columns:
                        contract = str(retention_customer_data['Contract'].iloc[0]).lower()
                        # Month-to-month typically has higher churn
                        if 'month' in contract:
                            churn_prob += 0.2
                        elif 'two year' in contract:
                            churn_prob -= 0.2

                    # Ensure probability is between 0 and 1
                    churn_prob = max(0.05, min(0.95, churn_prob))

                    # Generate recommendations
                    recommendations = generate_retention_recommendations(
                        churn_prob,
                        retention_customer_data,
                        type(st.session_state.model).__name__
                    )

                    # Display risk assessment
                    st.markdown("#### Churn Risk Assessment")

                    # Create a gauge chart for churn probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob * 100,
                        title={'text': "Churn Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ]
                        }
                    ))

                    fig.update_layout(
                        height=250,
                        plot_bgcolor='#0E1117',
                        paper_bgcolor='#0E1117',
                        font_color='white'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Risk Level:** {recommendations['risk_level']}")
                    st.markdown(recommendations['explanation'])

                    # Display immediate actions
                    st.markdown("#### Recommended Immediate Actions")
                    if recommendations['immediate_actions']:
                        for i, action in enumerate(recommendations['immediate_actions']):
                            st.markdown(f"{i+1}. {action}")
                    else:
                        st.markdown("No immediate actions required. Continue standard engagement.")

                    # Display strategic recommendations
                    st.markdown("#### Strategic Recommendations")
                    for i, strategy in enumerate(recommendations['strategic_recommendations']):
                        st.markdown(f"{i+1}. {strategy}")

                    # Additional retention tools
                    st.markdown("#### Retention Tools")

                    # Display different tools based on risk level
                    if recommendations['risk_level'] == "HIGH":
                        st.markdown("""
                        - **VIP Rescue Program**: Assign dedicated success manager for personalized attention
                        - **Exclusive Promotion**: Offer special time-limited discount or upgrade
                        - **Executive Outreach**: Schedule high-level contact for relationship building
                        - **Custom Package**: Design tailored solution for specific needs
                        """)
                    elif recommendations['risk_level'] == "MEDIUM":
                        st.markdown("""
                        - **Satisfaction Survey**: Identify and address specific pain points
                        - **Feature Education**: Ensure customer is utilizing all relevant features
                        - **Loyalty Rewards**: Provide additional value for continued loyalty
                        - **Usage Analysis**: Review usage patterns to suggest optimizations
                        """)
                    else:
                        st.markdown("""
                        - **Referral Program**: Leverage satisfaction to expand customer base
                        - **Upsell Opportunities**: Identify potential product/service expansions
                        - **Success Stories**: Share how similar customers are benefiting
                        - **Regular Check-ins**: Maintain engagement through periodic contact
                        """)

            else:
                st.info("Please train a model first to access retention recommendations.")

    else:
        st.info("Please load data and train a model to access AI-powered explanations.")
        st.markdown("""
        ### What You'll Get From AI Explanations

        Once you've trained a model, this tab will provide:

        1. **Model Explanations**: Understand what factors most influence customer churn
        2. **Customer Profile Analysis**: Get insights on individual customers and their churn risk
        3. **Retention Recommendations**: Receive personalized strategies to reduce churn

        Start by using the sample data, preprocessing it, and training a model in the previous tabs.
        """)

# Footer
st.markdown("---")
st.markdown("### About Customer Churn Prediction")
st.markdown("""
This application uses machine learning to predict which customers are likely to churn (leave your service).
By analyzing patterns in historical customer data, the model identifies key factors that contribute to churn 
and predicts future churn probability for existing or new customers.

**Key Benefits:**
- Identify at-risk customers before they leave
- Understand factors contributing to customer attrition
- Develop targeted retention strategies
- Improve customer satisfaction and loyalty
- Reduce customer acquisition costs
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ by S.Tamilselvan</p>
    <p>© 2023 Customer Churn Analysis and Prediction</p>
</div>
""", unsafe_allow_html=True)
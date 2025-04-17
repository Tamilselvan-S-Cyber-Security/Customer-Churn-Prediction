import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def handle_missing_values(df, strategy="Mean/Mode"):
    """
    Handle missing values in the dataframe

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    strategy : str
        Strategy to handle missing values ('Mean/Mode', 'Median/Mode', 'Drop Rows', 'Fill with Zero')

    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with missing values handled
    """
    if strategy == "Drop Rows":
        return df.dropna()

    # Create copies of dataframe for numeric and categorical columns
    df_numeric = df.select_dtypes(include=['number'])
    df_categorical = df.select_dtypes(exclude=['number'])

    # Handle numeric columns
    if not df_numeric.empty:
        if strategy == "Mean/Mode":
            imputer = SimpleImputer(strategy='mean')
        elif strategy == "Median/Mode":
            imputer = SimpleImputer(strategy='median')
        elif strategy == "Fill with Zero":
            imputer = SimpleImputer(strategy='constant', fill_value=0)

        df_numeric = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index
        )

    # Handle categorical columns
    if not df_categorical.empty:
        # Use mode imputation for categorical columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_categorical = pd.DataFrame(
            cat_imputer.fit_transform(df_categorical),
            columns=df_categorical.columns,
            index=df_categorical.index
        )

    # Combine the processed dataframes
    return pd.concat([df_numeric, df_categorical], axis=1)

def encode_categorical_features(df, strategy="One-Hot Encoding"):
    """
    Encode categorical features in the dataframe

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    strategy : str
        Encoding strategy ('One-Hot Encoding', 'Label Encoding')

    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with encoded categorical features
    """
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) == 0:
        return df

    if strategy == "One-Hot Encoding":
        # Apply one-hot encoding
        return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    elif strategy == "Label Encoding":
        df_encoded = df.copy()
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        return df_encoded

def apply_feature_scaling(df, method="StandardScaler"):
    """
    Apply feature scaling to numeric columns

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    method : str
        Scaling method ('StandardScaler', 'MinMaxScaler', 'RobustScaler')

    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with scaled features
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        return df

    # Select appropriate scaler
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()

    # Apply scaling
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_scaled

def remove_outliers_zscore(df, threshold=3.0):
    """
    Remove outliers using Z-score method

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    threshold : float
        Z-score threshold for outlier detection

    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with outliers removed
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        return df

    df_no_outliers = df.copy()

    for col in numeric_cols:
        # Calculate z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        # Keep only rows with z-scores below threshold
        df_no_outliers = df_no_outliers[z_scores < threshold]

    return df_no_outliers

def preprocess_data(df, options):
    """
    Preprocess the data based on selected options with enhanced CSV handling
    """
    try:
        # Make a copy and handle any string/object columns that should be numeric
        processed_df = df.copy()

        # Convert numeric columns that might be strings
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                try:
                    # Try to convert to numeric, fill errors with NaN
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                except:
                    continue

        # Remove excluded features
        if 'features_to_exclude' in options and options['features_to_exclude']:
            processed_df = processed_df.drop(columns=options['features_to_exclude'], errors='ignore')

        # Handle missing values
        if options.get('handle_missing', False):
            processed_df = handle_missing_values(processed_df, options.get('missing_strategy', 'Mean/Mode'))

        # Remove outliers
        if options.get('remove_outliers', False):
            processed_df = remove_outliers_zscore(processed_df, options.get('outlier_threshold', 3.0))

        # Encode categorical features
        if options.get('encode_categorical', False):
            processed_df = encode_categorical_features(processed_df, options.get('encoding_strategy', 'One-Hot Encoding'))

        # Apply feature scaling
        if options.get('feature_scaling', False):
            # Don't scale the target column
            if 'target_column' in options and options['target_column'] in processed_df.columns:
                target = processed_df[options['target_column']].copy()
                features_df = processed_df.drop(columns=[options['target_column']])
                features_df = apply_feature_scaling(features_df, options.get('scaling_method', 'StandardScaler'))
                # Recombine with target
                processed_df = pd.concat([features_df, target], axis=1)
            else:
                processed_df = apply_feature_scaling(processed_df, options.get('scaling_method', 'StandardScaler'))

        # Get feature names
        if 'target_column' in options and options['target_column'] in processed_df.columns:
            feature_names = [col for col in processed_df.columns if col != options['target_column']]
        else:
            feature_names = processed_df.columns.tolist()

        return processed_df, feature_names
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        return None, None


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to split
    target_column : str
        Name of the target column
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test
        Training and testing data splits
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test
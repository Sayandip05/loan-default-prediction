"""
Data preprocessing module
"""
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_params():
    """Load parameters from params.yaml"""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in the dataset
    
    Args:
        df: Input dataframe
        strategy: Strategy for filling missing values ('median', 'mean', 'mode')
    
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Handle MonthlyIncome missing values
    if strategy == 'median':
        df_clean['MonthlyIncome'].fillna(df_clean['MonthlyIncome'].median(), inplace=True)
        df_clean['NumberOfDependents'].fillna(df_clean['NumberOfDependents'].median(), inplace=True)
    elif strategy == 'mean':
        df_clean['MonthlyIncome'].fillna(df_clean['MonthlyIncome'].mean(), inplace=True)
        df_clean['NumberOfDependents'].fillna(df_clean['NumberOfDependents'].mean(), inplace=True)
    
    return df_clean


def handle_outliers(df):
    """
    Handle outliers in the dataset
    
    Args:
        df: Input dataframe
    
    Returns:
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    
    # Handle age = 0
    df_clean['age'].replace(0, df_clean['age'].median(), inplace=True)
    
    # Handle extreme outliers in RevolvingUtilizationOfUnsecuredLines
    Q1 = df_clean['RevolvingUtilizationOfUnsecuredLines'].quantile(0.25)
    Q3 = df_clean['RevolvingUtilizationOfUnsecuredLines'].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 3 * IQR
    
    df_clean.loc[
        df_clean['RevolvingUtilizationOfUnsecuredLines'] > upper_limit,
        'RevolvingUtilizationOfUnsecuredLines'
    ] = upper_limit
    
    # Cap extreme debt ratios
    df_clean.loc[df_clean['DebtRatio'] > 5000, 'DebtRatio'] = 5000
    
    return df_clean


def preprocess_data(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Main preprocessing function
    
    Args:
        input_path: Path to raw data CSV
        output_dir: Directory to save processed data
        test_size: Test set proportion
        random_state: Random seed
    """
    print("Loading data...")
    df = pd.read_csv(input_path, index_col=0)
    print(f"Original shape: {df.shape}")
    
    print("Handling missing values...")
    df_clean = handle_missing_values(df, strategy='median')
    
    print("Handling outliers...")
    df_clean = handle_outliers(df_clean)
    
    print("Splitting data...")
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df_clean.drop('SeriousDlqin2yrs', axis=1)
    y = df_clean['SeriousDlqin2yrs']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Combine back for saving
    train_data = X_train.copy()
    train_data['SeriousDlqin2yrs'] = y_train
    
    test_data = X_test.copy()
    test_data['SeriousDlqin2yrs'] = y_test
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"✅ Preprocessing complete!")
    print(f"Train data saved to: {train_path} (shape: {train_data.shape})")
    print(f"Test data saved to: {test_path} (shape: {test_data.shape})")
    print(f"Class distribution in train:")
    print(train_data['SeriousDlqin2yrs'].value_counts(normalize=True))


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    preprocess_params = params['preprocess']
    
    # Define paths
    input_path = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    # Run preprocessing
    preprocess_data(
        input_path=str(input_path),
        output_dir=str(output_dir),
        test_size=preprocess_params['test_size'],
        random_state=preprocess_params['random_state']
    )

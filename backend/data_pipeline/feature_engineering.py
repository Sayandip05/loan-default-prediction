"""
Feature engineering module
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


def create_debt_income_ratio(df):
    """Create debt to income ratio feature"""
    df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    return df


def create_credit_utilization_bins(df):
    """Create credit utilization category"""
    df['CreditUtilization_Category'] = pd.cut(
        df['RevolvingUtilizationOfUnsecuredLines'],
        bins=[0, 0.3, 0.6, 1.0, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def create_age_bins(df):
    """Create age group feature"""
    df['AgeGroup'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def create_past_due_features(df):
    """Create past due aggregated features"""
    # Total past due events
    df['TotalPastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    # Has any past due
    df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)
    
    return df


def create_income_features(df):
    """Create income-related features"""
    # Income per dependent
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    
    # Log transform for income (handle skewness)
    df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    
    return df


def create_credit_features(df):
    """Create credit-related features"""
    # Loans per credit line
    df['LoansPerCreditLine'] = df['NumberRealEstateLoansOrLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    
    return df


def engineer_features(input_train_path, input_test_path, output_dir):
    """
    Main feature engineering function
    
    Args:
        input_train_path: Path to preprocessed train data
        input_test_path: Path to preprocessed test data
        output_dir: Directory to save feature-engineered data
    """
    print("Loading preprocessed data...")
    train_df = pd.read_csv(input_train_path)
    test_df = pd.read_csv(input_test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Load parameters
    params = load_params()
    fe_params = params['feature_engineering']
    
    # Apply feature engineering
    print("Creating features...")
    
    for df in [train_df, test_df]:
        if fe_params.get('create_debt_income_ratio', True):
            df = create_debt_income_ratio(df)
        
        if fe_params.get('create_credit_utilization_bins', True):
            df = create_credit_utilization_bins(df)
        
        if fe_params.get('create_age_bins', True):
            df = create_age_bins(df)
        
        df = create_past_due_features(df)
        df = create_income_features(df)
        df = create_credit_features(df)
    
    # Save engineered data
    os.makedirs(output_dir, exist_ok=True)
    train_output_path = os.path.join(output_dir, 'train_features.csv')
    test_output_path = os.path.join(output_dir, 'test_features.csv')
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"✅ Feature engineering complete!")
    print(f"Train features saved to: {train_output_path} (shape: {train_df.shape})")
    print(f"Test features saved to: {test_output_path} (shape: {test_df.shape})")
    print(f"\nNew features created: {train_df.shape[1] - pd.read_csv(input_train_path).shape[1]}")


if __name__ == "__main__":
    # Define paths
    input_train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    input_test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    # Run feature engineering
    engineer_features(
        input_train_path=str(input_train_path),
        input_test_path=str(input_test_path),
        output_dir=str(output_dir)
    )

"""
Prediction module
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Union


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class LoanDefaultPredictor:
    """Loan Default Prediction Class"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model
        """
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "model.pkl"
        
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data
        
        Args:
            data: Input data as dict or DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Apply same feature engineering as training
        df = self._engineer_features(df)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (same as training)"""
        
        # Debt to income ratio
        df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
        
        # Credit utilization category
        df['CreditUtilization_Category'] = pd.cut(
            df['RevolvingUtilizationOfUnsecuredLines'],
            bins=[0, 0.3, 0.6, 1.0, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Age group
        df['AgeGroup'] = pd.cut(
            df['age'],
            bins=[0, 30, 45, 60, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Total past due
        df['TotalPastDue'] = (
            df['NumberOfTime30-59DaysPastDueNotWorse'] +
            df['NumberOfTime60-89DaysPastDueNotWorse'] +
            df['NumberOfTimes90DaysLate']
        )
        
        # Has past due
        df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)
        
        # Income per dependent
        df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
        
        # Log monthly income
        df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])
        
        # Loans per credit line
        df['LoansPerCreditLine'] = df['NumberRealEstateLoansOrLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
        
        return df
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data: Input data
            
        Returns:
            Predictions (0 or 1)
        """
        df = self.preprocess_input(data)
        predictions = self.model.predict(df)
        return predictions
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            data: Input data
            
        Returns:
            Prediction probabilities
        """
        df = self.preprocess_input(data)
        probabilities = self.model.predict_proba(df)
        return probabilities
    
    def predict_single(self, data: Dict) -> Dict:
        """
        Predict for single input and return detailed result
        
        Args:
            data: Single input as dictionary
            
        Returns:
            Dictionary with prediction and probability
        """
        prediction = self.predict(data)[0]
        proba = self.predict_proba(data)[0]
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'probability_no_default': float(proba[0]),
            'probability_default': float(proba[1]),
            'risk_level': self._get_risk_level(proba[1])
        }
    
    def _get_risk_level(self, default_prob: float) -> str:
        """Categorize risk level based on default probability"""
        if default_prob < 0.3:
            return 'Low Risk'
        elif default_prob < 0.6:
            return 'Medium Risk'
        else:
            return 'High Risk'


if __name__ == "__main__":
    # Test prediction
    predictor = LoanDefaultPredictor()
    
    # Sample input
    sample_data = {
        'RevolvingUtilizationOfUnsecuredLines': 0.766127,
        'age': 45,
        'NumberOfTime30-59DaysPastDueNotWorse': 2,
        'DebtRatio': 0.802982,
        'MonthlyIncome': 9120,
        'NumberOfOpenCreditLinesAndLoans': 13,
        'NumberOfTimes90DaysLate': 0,
        'NumberRealEstateLoansOrLines': 6,
        'NumberOfTime60-89DaysPastDueNotWorse': 0,
        'NumberOfDependents': 2
    }
    
    result = predictor.predict_single(sample_data)
    print("\n🔮 Prediction Result:")
    print(f"  Prediction: {result['prediction_label']}")
    print(f"  Default Probability: {result['probability_default']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")

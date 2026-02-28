"""
Model Training with MLflow Tracking
"""
import pandas as pd
import numpy as np
import yaml
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_params():
    """Load parameters from params.yaml"""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_data(train_path):
    """Load training data"""
    print(f"Loading data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Separate features and target
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    
    print(f"Data loaded: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y


def handle_imbalance(X_train, y_train, strategy='SMOTE'):
    """Handle class imbalance"""
    if strategy == 'SMOTE':
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_resampled.shape}")
        print(f"Class distribution:\n{pd.Series(y_resampled).value_counts(normalize=True)}")
        return X_resampled, y_resampled
    return X_train, y_train


def get_model(algorithm, params):
    """Get model based on algorithm"""
    if algorithm == 'xgboost':
        return XGBClassifier(**params['xgboost'])
    elif algorithm == 'random_forest':
        return RandomForestClassifier(**params['random_forest'])
    elif algorithm == 'logistic_regression':
        return LogisticRegression(**params['logistic_regression'])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    return metrics


def train_model(train_path, output_dir, params):
    """Main training function with MLflow tracking"""
    
    # Load data
    X, y = load_data(train_path)
    
    # Split data
    train_params = params['train']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_params['test_size'],
        random_state=train_params['random_state'],
        stratify=y
    )
    
    # Handle imbalance
    if train_params.get('handle_imbalance', False):
        X_train, y_train = handle_imbalance(
            X_train, y_train,
            strategy=train_params.get('imbalance_strategy', 'SMOTE')
        )
    
    # Set MLflow experiment
    mlflow_params = params['mlflow']
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    mlflow.set_experiment(mlflow_params['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Get model
        algorithm = params['model']['algorithm']
        print(f"\nTraining {algorithm} model...")
        model = get_model(algorithm, params['model'])
        
        # Log parameters
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_params(params['model'][algorithm])
        mlflow.log_param("test_size", train_params['test_size'])
        mlflow.log_param("handle_imbalance", train_params['handle_imbalance'])
        
        # Train model
        model.fit(X_train, y_train)
        print("✅ Training complete!")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=train_params.get('cv_folds', 5),
            scoring='roc_auc'
        )
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\n📊 Test Set Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")
            mlflow.log_metric(metric_name, metric_value)
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🎯 Top 10 Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            # Save feature importance
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=mlflow_params['model_name']
        )
        
        # Save metrics to JSON
        metrics_path = PROJECT_ROOT / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Metrics saved to: {metrics_path}")
        print(f"\n🎉 Training complete! MLflow run: {mlflow.active_run().info.run_id}")
        
        return model, metrics


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    
    # Define paths
    train_path = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    output_dir = PROJECT_ROOT / "models"
    
    # Train model
    model, metrics = train_model(
        train_path=str(train_path),
        output_dir=str(output_dir),
        params=params
    )

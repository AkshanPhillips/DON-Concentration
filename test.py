import torch
import torch.nn as nn
import numpy as np
from data_processing import load_and_preprocess_data, augment_data, visualize_data
from models import SpectralCNN, AttentionCNN
from train import train_model, optimize_hyperparameters
from evaluation import evaluate_model, stratified_error_analysis, shap_analysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_pipeline():
    """Runs an end-to-end test of the full CNN pipeline with comparison."""
    print("Loading and preprocessing data...")
    file_path = "/home/geekbull/Downloads/MLE-Assignment - MLE-Assignment.csv"
    X, y, _, df = load_and_preprocess_data(file_path)
    
    visualize_data(df)
    X_aug, y_aug = augment_data(X, y)
    logger.info(f"Augmented dataset size: {X_aug.shape[0]} samples")
    
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X_aug, y_aug)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    models = {
        "SpectralCNN": SpectralCNN(input_length=448),
        "AttentionCNN": AttentionCNN(input_length=448),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} model...")
        if name != "RandomForest":
            trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs=500, 
                                        lr=best_params['learning_rate'], batch_size=best_params['batch_size'])
            y_pred_log = evaluate_model(trained_model, X_test, y_test)
        else:
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            y_pred_log = model.predict(X_test.reshape(X_test.shape[0], -1))
        
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred_log)
        y_pred_orig = np.clip(y_pred_orig, 0, 50000)
        
        results[name] = {
            "Log MAE": mean_absolute_error(y_test, y_pred_log),
            "Log RMSE": np.sqrt(mean_squared_error(y_test, y_pred_log)),
            "Log R²": r2_score(y_test, y_pred_log),
            "Orig MAE": mean_absolute_error(y_test_orig, y_pred_orig),
            "Orig RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            "Orig R²": r2_score(y_test_orig, y_pred_orig)
        }
        if name != "RandomForest":
            stratified_error_analysis(y_test, y_pred_log)
            shap_analysis(trained_model, X_test)
    
    # Comparison Table
    print("\nModel Comparison Table:")
    print("| Model          | Log MAE | Log RMSE | Log R² | Orig MAE | Orig RMSE | Orig R² |")
    print("|----------------|---------|----------|--------|----------|-----------|---------|")
    for name, metrics in results.items():
        print(f"| {name:<14} | {metrics['Log MAE']:.4f} | {metrics['Log RMSE']:.4f} | {metrics['Log R²']:.4f} | "
              f"{metrics['Orig MAE']:.1f} | {metrics['Orig RMSE']:.1f} | {metrics['Orig R²']:.4f} |")
    
    torch.save(models["AttentionCNN"].state_dict(), "best_attention_cnn_model.pth")
    print("Pipeline test completed successfully!")

if __name__ == "__main__":
    test_full_pipeline()
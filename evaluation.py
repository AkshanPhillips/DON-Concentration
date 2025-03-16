import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cpu")  # Force CPU for consistency



def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).numpy().flatten()
    return y_pred

def stratified_error_analysis(y_true, y_pred, n_bins=5):
    bins = np.linspace(min(y_true), max(y_true), n_bins + 1)
    bin_indices = np.digitize(y_true, bins[:-1])
    bin_results = []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            bin_results.append({
                "Bin Range": f"{bins[i-1]:.2f}-{bins[i]:.2f}",
                "MAE": bin_mae,
                "Samples": np.sum(mask)
            })
            logger.info(f"Bin {i} ({bins[i-1]:.2f}-{bins[i]:.2f}): MAE = {bin_mae:.4f}, Samples = {np.sum(mask)}")
    return bin_results
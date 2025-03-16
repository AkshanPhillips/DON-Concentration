import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import optuna
import logging
import numpy as np
from models import SpectralCNN  # Ensure this import matches your structure
from torch.utils.data import DataLoader, TensorDataset


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cpu")  # Force CPU for consistency
logger.info(f"Using device: {device}")




def hybrid_loss(y_pred, y_true, gamma=2.0, low_threshold=2.34):
    error = (y_pred - y_true) ** 2
    focal_weight = torch.abs(y_true - y_pred) ** gamma
    focal_loss = torch.mean(focal_weight * error)
    low_mask = (y_true < low_threshold).float()
    low_weight = 100.0 * low_mask + 1.0 * (1 - low_mask)  # Reduced from 200.0
    mse_loss = torch.mean(low_weight * error)
    return 0.5 * focal_loss + 0.5 * mse_loss

def train_model(model, X_train, y_train, X_val, y_val, epochs=500, lr=0.0001, batch_size=16):
    """Trains the CNN with mini-batches, validation, and early stopping."""
    criterion = hybrid_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                             torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    best_val_mae = float('inf')
    patience, trigger = 100, 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).numpy().flatten()
            val_mae = mean_absolute_error(y_val, val_outputs)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val MAE: {val_mae:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}_model.pth")
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

# Custom Training Function for SpectralCNN
def train_spectral_cnn(model, X_train, y_train, X_val, y_val, epochs=300, lr=0.0001, batch_size=16):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping_patience = 50
    best_mae = float('inf')
    patience_counter = 0

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = hybrid_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor).flatten()
            val_mae = mean_absolute_error(y_val, y_val_pred.numpy())

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f"best_{model.__class__.__name__}.pth"))
    return model

def optimize_hyperparameters(X, y, n_trials=20):
    """Optimizes CNN hyperparameters using Optuna with 5-fold CV."""
    logger.info(f"Optimizing hyperparameters with {n_trials} trials")

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        low_weight = trial.suggest_float("low_weight", 50.0, 300.0, step=50.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = SpectralCNN(input_length=448)
            model = train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=lr, batch_size=batch_size)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            with torch.no_grad():
                y_pred = model(X_val_tensor).numpy().flatten()
            mae_scores.append(mean_absolute_error(y_val, y_pred))
        
        return np.mean(mae_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best hyperparameters: {study.best_params}, Best MAE: {study.best_value:.4f}")
    return study.best_params

if __name__ == "__main__":
    from data_processing import load_and_preprocess_data, augment_data
    file_path = "/home/geekbull/Downloads/MLE-Assignment - MLE-Assignment.csv"
    X, y, _, _ = load_and_preprocess_data(file_path)
    X_aug, y_aug = augment_data(X, y)
    
    best_params = optimize_hyperparameters(X_aug, y_aug)
    model = SpectralCNN(input_length=448)
    X_temp, X_test, y_temp, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs=500, 
                                lr=best_params['learning_rate'], batch_size=best_params['batch_size'])
    torch.save(trained_model.state_dict(), "best_cnn_model.pth")
    logger.info("CNN model trained and saved successfully.")
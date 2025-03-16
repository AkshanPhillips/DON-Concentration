import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_and_preprocess_data, augment_data, visualize_data
from models import SpectralCNN, AttentionCNN
from train import train_model, optimize_hyperparameters, hybrid_loss
from evaluation import evaluate_model, stratified_error_analysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import logging
import io
import shap
import streamlit as st
import gc
import psutil

# Set up logging
log_buffer = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(log_buffer), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Corn DON Concentration Prediction", layout="wide")
st.title("Corn DON Concentration Prediction Pipeline")
st.write("Upload hyperspectral data to explore, train, and predict DON concentration (ppb).")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
run_pipeline = st.sidebar.button("Run Full Pipeline")
skip_shap = st.sidebar.checkbox("Skip SHAP Analysis (saves memory)", value=True)

device = torch.device("cpu")  
logger.info(f"Using device: {device}")

def display_logs():
    log_contents = log_buffer.getvalue()
    st.subheader("Logs")
    st.text_area("Runtime Logs", log_contents, height=200)

@st.cache_data
def process_data(file):
    if file is not None:
        X, y, scaler, df = load_and_preprocess_data(file)
        return X, y, scaler, df
    return None, None, None, None

def display_dataset_info(df):
    st.subheader("Dataset Information")
    st.write(f"**Shape**: {df.shape}")
    st.write(f"**Number of Samples**: {df.shape[0]}")
    st.write(f"**Number of Features**: {df.shape[1] - 2}")
    missing_values = df.isnull().sum()
    st.write("**Missing Values by Column**:")
    st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
    st.write("**Target Variable (DON Concentration) Statistics**:")
    st.write(df['vomitoxin_ppb'].describe())

def display_visualizations(df):
    st.subheader("Data Visualizations")
    visualize_data(df)
    fig = px.histogram(df, x='vomitoxin_ppb', nbins=30, title='Distribution of DON Concentration (ppb)')
    st.plotly_chart(fig, use_container_width=True)
    spectral_columns = [str(i) for i in range(448)]
    mean_spectrum = df[spectral_columns].mean()
    wavelengths = np.linspace(400, 2500, len(spectral_columns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=mean_spectrum, mode='lines', name='Mean Spectral Signature'))
    fig.update_layout(title='Mean Spectral Signature', xaxis_title='Wavelength (nm)', yaxis_title='Reflectance')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(df, y='vomitoxin_ppb', title='Box Plot of DON Concentration')
    st.plotly_chart(fig, use_container_width=True)
    sample_subset = df.iloc[:20, 1:100]
    fig = px.imshow(sample_subset.values, labels=dict(x="Wavelength Index", y="Sample Index", color="Reflectance"),
                    title="Heatmap of Spectral Reflectance (First 20 Samples, First 100 Wavelengths)")
    st.plotly_chart(fig, use_container_width=True)

def plot_model_performance(model_name, y_test, y_pred_log):
    st.subheader(f"Model Performance (Log Scale) - {model_name}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_log, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual DON (log scale)')
    ax.set_ylabel('Predicted DON (log scale)')
    ax.set_title(f'Actual vs. Predicted DON (Log Scale) - {model_name}')
    st.pyplot(fig)
    fig, ax = plt.subplots()
    residuals = y_test - y_pred_log
    ax.scatter(y_pred_log, residuals, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted DON (log scale)')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot (Log Scale) - {model_name}')
    st.pyplot(fig)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory Usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

def run_training_and_evaluation(X, y):
    st.subheader("Training and Evaluation")
    X_aug, y_aug = augment_data(X, y, n_augmented=400)  
    st.write(f"Augmented dataset size: {X_aug.shape[0]} samples")
    log_memory_usage()

    with st.spinner("Optimizing hyperparameters..."):
        best_params = optimize_hyperparameters(X_aug, y_aug, n_trials=5)  
        st.write("Best Hyperparameters:", best_params)

    X_temp, X_test, y_temp, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    st.write(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    log_memory_usage()

    models = {
        "LinearRegression": LinearRegression(),
        "SpectralCNN": SpectralCNN(input_length=448).to(device),
        "AttentionCNN": AttentionCNN(input_length=448).to(device),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    }
    results = {}

    for name, model in models.items():
        st.write(f"Training {name}...")
        logger.info(f"Starting training for {name}")
        log_memory_usage()
        try:
            if name in ["SpectralCNN", "AttentionCNN"]:
                logger.info(f"Initializing {name} model")
                test_input = torch.randn(1, 1, 448).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                logger.info(f"{name} forward pass test successful, output shape: {test_output.shape}")

                logger.info(f"Training {name} with {X_train.shape[0]} samples")
                trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs=500,
                                            lr=best_params['learning_rate'], batch_size=best_params['batch_size'])
                logger.info(f"Evaluating {name}")
                y_pred_log = evaluate_model(trained_model, X_test, y_test)
            else:
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                logger.info(f"Training {name} with {X_train_2d.shape[0]} samples")
                if name == "XGBoost":
                    # Basic tuning for XGBoost
                    param_grid = {
                        'n_estimators': [200, 300],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                    best_score = float('inf')
                    best_params_xgb = {}
                    for n_est in param_grid['n_estimators']:
                        for depth in param_grid['max_depth']:
                            for lr in param_grid['learning_rate']:
                                xgb_model = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42)
                                xgb_model.fit(X_train_2d, y_train)
                                y_pred = xgb_model.predict(X_test_2d)
                                score = mean_absolute_error(y_test, y_pred)
                                if score < best_score:
                                    best_score = score
                                    best_params_xgb = {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr}
                    model = xgb.XGBRegressor(**best_params_xgb, random_state=42)
                    logger.info(f"Best XGBoost params: {best_params_xgb}")
                model.fit(X_train_2d, y_train)
                y_pred_log = model.predict(X_test_2d)

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
            plot_model_performance(name, y_test, y_pred_log)
            st.subheader(f"Stratified Error Analysis - {name}")
            bin_results = stratified_error_analysis(y_test, y_pred_log, n_bins=5)
            st.table(pd.DataFrame(bin_results))
            logger.info(f"Completed training and evaluation for {name}")
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            st.error(f"Error training {name}: {str(e)}")
            continue

        del y_pred_log, y_pred_orig
        gc.collect()
        log_memory_usage()

    st.subheader("Test Set Predictions - SpectralCNN")
    y_test_pred_log = evaluate_model(models["SpectralCNN"], X_test, y_test)
    y_test_pred = np.expm1(y_test_pred_log)
    comparison_df = pd.DataFrame({
        "Actual DON (ppb)": np.expm1(y_test),
        "Predicted DON (ppb)": np.clip(y_test_pred, 0, 50000)
    })
    st.table(comparison_df.head(10))

    st.subheader("Model Comparison")
    table_data = pd.DataFrame(results).T
    st.table(table_data)

    if not skip_shap:
        st.subheader("SHAP Analysis - SpectralCNN")
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.KernelExplainer(
                    lambda x: models["SpectralCNN"](torch.tensor(x.reshape(-1, 1, 448), dtype=torch.float32, device=device)).detach().numpy().flatten(),
                    X_test[:5].reshape(5, -1)
                )
                shap_values = explainer.shap_values(X_test[:1].reshape(1, -1))
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test[:1].reshape(1, -1), feature_names=[f"W{i}" for i in range(448)], show=False)
                st.pyplot(fig)
                logger.info("SHAP values computed and plotted")
            except MemoryError:
                st.warning("SHAP computation skipped due to memory constraints.")
                logger.warning("MemoryError during SHAP computation")

    gc.collect()
    return models["SpectralCNN"]

def predict(model, X_input):
    X_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(X_tensor).numpy().flatten()
    return np.expm1(prediction)

if uploaded_file and run_pipeline:
    try:
        with st.spinner("Processing data..."):
            X, y, scaler, df = process_data(uploaded_file)
            if X is not None:
                display_dataset_info(df)
                display_visualizations(df)
                trained_model = run_training_and_evaluation(X, y)
                st.subheader("Predictions on Uploaded Data")
                predictions = predict(trained_model, X)
                st.write("Predicted DON Concentrations (ppb):", predictions[:10].tolist(), "... (showing first 10)")
    except MemoryError:
        logger.error("Out of memory error occurred during pipeline execution")
        st.error("Pipeline failed due to insufficient memory. Try skipping SHAP or reducing dataset size.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
    display_logs()
elif uploaded_file:
    st.write("Click 'Run Full Pipeline' to process the uploaded file.")
else:
    st.write("Please upload a CSV file to begin.")

st.sidebar.subheader("Instructions")
st.sidebar.write("""
1. Upload a CSV file with hyperspectral data.
2. Click 'Run Full Pipeline' to process the data, train the model, and view results.
3. Check 'Skip SHAP Analysis' to save memory if needed.
4. Explore visualizations, metrics, and predictions in the main panel.
""")
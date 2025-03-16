import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """Loads and preprocesses hyperspectral data for DON concentration prediction."""
    df = pd.read_csv(file_path)
    logger.info("Dataset loaded successfully")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of samples: {df.shape[0]}")
    logger.info(f"Number of features: {df.shape[1] - 2}")
    logger.info(f"Column names (first 10): {list(df.columns[:10])}...")

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    missing_by_column = df.isnull().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0]
    if len(columns_with_missing) > 0:
        logger.warning(f"Columns with missing values: {columns_with_missing.to_dict()}")
    else:
        logger.info("No missing values found in any column")

    target_stats = df['vomitoxin_ppb'].describe()
    logger.info("Target variable (DON concentration) statistics:")
    logger.info(target_stats)

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df.iloc[:, 1:-1])
    y = np.log1p(np.clip(df['vomitoxin_ppb'].values, 0, 50000))  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_reshaped = X_scaled.reshape(-1, 1, 448)

    return X_reshaped, y, scaler, df

def augment_data(X, y, n_augmented=400):
    noise = np.random.normal(0, 0.02, (n_augmented, 1, 448))
    scaling = np.random.uniform(0.95, 1.05, (n_augmented, 1, 1))
    low_idx = np.where(y < 2.34)[0]
    if len(low_idx) > 0:
        low_sample_idx = np.random.choice(low_idx, int(n_augmented * 0.7), replace=True)  
        other_idx = np.random.choice(len(y), n_augmented - len(low_sample_idx), replace=True)
        sample_idx = np.concatenate([low_sample_idx, other_idx])
    else:
        sample_idx = np.random.choice(len(y), n_augmented, replace=True)
    X_subset = X[sample_idx]
    X_augmented = (X_subset + noise) * scaling
    X_aug = np.concatenate([X, X_augmented])
    y_aug = np.concatenate([y, y[sample_idx]])
    return X_aug, y_aug

def visualize_data(df):
    """Generates exploratory visualizations for the dataset."""
    fig = px.histogram(df, x='vomitoxin_ppb', nbins=30,
                       title='Distribution of DON Concentration (vomitoxin_ppb)')
    fig.update_layout(xaxis_title='DON Concentration (ppb)', yaxis_title='Count', plot_bgcolor='white')
    fig.show()

    spectral_columns = [str(i) for i in range(448)]
    mean_spectrum = df[spectral_columns].mean()
    wavelengths = np.linspace(400, 2500, len(spectral_columns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=mean_spectrum, mode='lines', name='Mean Spectral Signature'))
    fig.update_layout(title='Mean Spectral Signature Across All Samples',
                      xaxis_title='Wavelength (nm)', yaxis_title='Reflectance', plot_bgcolor='white')
    fig.show()

    fig = px.box(df, y='vomitoxin_ppb', title='Box Plot of DON Concentration')
    fig.update_layout(yaxis_title='DON Concentration (ppb)', plot_bgcolor='white')
    fig.show()

    sample_subset = df.iloc[:20, 1:100]
    fig = px.imshow(sample_subset.values,
                    labels=dict(x="Wavelength Index", y="Sample Index", color="Reflectance"),
                    title="Heatmap of Spectral Reflectance (First 20 Samples, First 100 Wavelengths)")
    fig.update_layout(plot_bgcolor='white')
    fig.show()

if __name__ == "__main__":
    file_path = "/home/geekbull/Downloads/MLE-Assignment - MLE-Assignment.csv"
    X, y, scaler, df = load_and_preprocess_data(file_path)
    visualize_data(df)
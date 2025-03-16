# DON-Concentration

Key Points
This README file is for a GitHub repository containing a machine learning pipeline to predict DON (mycotoxin) levels in corn using hyperspectral data.
It includes data processing, model training, evaluation, and a web application for interactive use.
Setup involves cloning the repository, setting up a virtual environment, and installing dependencies from requirements.txt.

Project Overview
This repository provides a complete machine learning pipeline for predicting Deoxynivalenol (DON) levels in corn samples using hyperspectral imaging data. It is designed to be production-ready, with modular code, testing, and documentation, suitable for agricultural and food safety applications.

Setup Instructions
To get started, clone the repository and set up your environment:

Create and activate a virtual environment with python -m venv venv and source venv/bin/activate.
Install dependencies by running pip install -r requirements.txt, ensuring libraries like PyTorch, Scikit-learn, and Streamlit are included.
Running the Code
You can run individual components or the entire pipeline:

Process data with python data_processing.py.
Train models with python: train.py.
Evaluate models with python: evaluation.py.
Test the full pipeline with python: test.py.
For interactive use, run the Streamlit app with streamlit run: App.py and access it at http://localhost:8501.

Project Background and Objective
The repository addresses a critical need in agricultural and food safety by predicting DON concentration in corn using hyperspectral imaging data. The objective, as specified, is to develop a machine learning pipeline that includes data preprocessing, model training, evaluation, and deployment, with a focus on modularity and production readiness. This is particularly relevant for ensuring corn sample safety, given DON's toxicity.

The dataset consists of spectral reflectance values across 448 wavelength bands (400 nm to 2500 nm) for each corn sample, with the target variable being DON concentration in parts per billion (ppb). This setup is ideal for regression tasks, leveraging the rich spectral information for accurate predictions.

Repository Structure and File Descriptions
The repository is organized into several Python files, each serving a specific purpose within the pipeline:

data_processing.py: This file handles the initial data loading and preprocessing steps. It loads hyperspectral data from a CSV file, imputes missing values using median strategy, scales features with StandardScaler, and transforms the target variable using np.log1p to handle skewness, capping outliers at 50,000 ppb. It also includes data augmentation to balance the dataset, focusing on low DON concentrations, and generates visualizations like histograms, line plots, box plots, and heatmaps using Plotly and Matplotlib. The script is designed to work with a specific file path, which users must update for their data.
models.py: This file defines two active CNN models: SpectralCNN and AttentionCNN, both for 1D spectral data processing. SpectralCNN is a simpler CNN with convolutional layers, max pooling, and fully connected layers, while AttentionCNN incorporates a multi-head attention mechanism for feature focus. A third model, TransformerCNN, is defined but commented out, indicating potential future expansion. These models are implemented in PyTorch, suitable for regression tasks.
train.py: This script contains functions for training the models, using a custom hybrid loss function combining focal loss and weighted mean squared error (MSE), with parameters like gamma=2.0 and low_weight=100.0 to emphasize low DON values. It implements hyperparameter optimization using Optuna, searching over learning rate (1e-5 to 1e-3), batch size (8, 16, 32), and loss weights. Data is split into 64% training, 16% validation, and 20% test sets, with early stopping (patience 50-100 epochs) and logging for progress tracking. The training runs on CPU, as specified by torch.device('cpu').
evaluation.py: This file includes functions for model evaluation, specifically evaluate_model for making predictions and stratified_error_analysis for calculating Mean Absolute Error (MAE) across bins of true values. While it imports visualization libraries like Matplotlib and Seaborn, no plots are implemented, focusing instead on metrics like MAE, with potential for RMSE and R² (noted in commented code). SHAP is imported but not used, suggesting possible future interpretability additions.
test.py: This script serves as an end-to-end test, running the full pipeline from data processing to model evaluation. It tests SpectralCNN, AttentionCNN, and traditional models like Random Forest, comparing performances using MAE, RMSE, and R² in both log and original scales. It logs progress and saves the best model, making it a comprehensive validation of the system.
App.py: This is a Streamlit web application for interactive use, allowing users to upload CSV files with hyperspectral data, process it, train multiple models (Linear Regression, Random Forest, XGBoost, SpectralCNN, AttentionCNN), and view results. It includes visualizations (histograms, spectral signatures, scatter plots), error analysis, and optional SHAP analysis for interpretability. Users can control the pipeline via a sidebar, with real-time logging and memory management to handle large datasets.

File Name	Purpose
data_processing.py	Loads, preprocesses, augments, and visualizes hyperspectral data.
models.py	Defines CNN models (SpectralCNN, AttentionCNN) for regression.
train.py	Trains models with hybrid loss, optimizes hyperparameters using Optuna.
evaluation.py	Evaluates models, performs stratified error analysis.
test.py	Runs end-to-end pipeline, tests system comprehensively.
App.py	Streamlit app for interactive data upload, analysis, and prediction.
Another table for key dependencies:

Dependency	Purpose
PyTorch	Deep learning model implementation.
Scikit-learn	Traditional ML models, data splitting, metrics.
Matplotlib	Data visualization.
Seaborn	Enhanced statistical visualizations.
Plotly	Interactive plots in web application.
Streamlit	Web interface for interactive analysis.
Optuna	Hyperparameter optimization.
SHAP	Model interpretability (optional in App.py).
Conclusion
This README file ensures users can set up, run, and understand the machine learning pipeline for DON prediction, with detailed instructions for both developers and end-users. The inclusion of a Streamlit app enhances accessibility, while the modular code structure supports production readiness, aligning with the project's goals.

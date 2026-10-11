w# Kaggle Behavioral Fraud Detection

This project provides a starter structure for a fraud detection workflow with MLflow tracking.

## Project structure

- data/raw/ - raw CSV files from Kaggle
- data/processed/ - cleaned and engineered datasets
- notebooks/ - EDA and experimentation notebooks
- src/data/ - data loading and preprocessing scripts
- src/models/ - training and evaluation scripts
- src/utils/ - utility helpers
- configs/ - configuration files
- mlruns/ - MLflow experiment artifacts

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your Kaggle dataset in data/raw/.

3. Run the baseline training script:
   ```bash
   python src/models/train.py
   ```

4. Start the MLflow UI:
   ```bash
   mlflow ui
   ```

Then open the local MLflow UI in your browser.

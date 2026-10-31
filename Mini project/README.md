# Waste Generation Prediction

A beginner-friendly machine-learning mini project that estimates waste quantity from input features using **Linear Regression**.

## Features
- Loads a CSV dataset (`waste.csv`)
- Encodes categorical features automatically
- Trains and evaluates a Linear Regression model
- Displays MSE, MAE, and R² metrics
- Provides a simple Streamlit interface for predictions
- Shows a comparison chart for actual and predicted values

## Project structure
```text
waste-generation-prediction/
├── app.py
├── waste.csv
├── requirements.txt
└── README.md
```

## Dataset
The included `waste.csv` is a **synthetic sample dataset for demonstration**. Its values are illustrative and should not be treated as real municipal waste statistics.

The target column is `Waste_Quantity`. All other columns are used as model input features.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. (Optional) Create and activate a virtual environment.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the application:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL shown in the terminal.

## Upload to GitHub
1. Create a new repository on GitHub, for example `waste-generation-prediction`.
2. Upload `app.py`, `waste.csv`, `requirements.txt`, and `README.md`.
3. Add a short description: `Waste generation prediction using Linear Regression and Streamlit`.

## Model notes
- The dataset is small, so the displayed metrics are only a demonstration.
- Model performance depends on the quality, size, and representativeness of the dataset.
- Predictions should not be used for official waste-management decisions without validation on real data.

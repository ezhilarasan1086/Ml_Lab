"""
Waste Generation Prediction — mini project
Run: streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Waste Generation Prediction", page_icon="♻️", layout="wide")
st.title("♻️ Waste Generation Prediction")
st.write("Predict the quantity of waste generated using basic demographic and environmental inputs.")

DATA_PATH = Path(__file__).parent / "waste.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["Waste_Quantity"])
    y = data["Waste_Quantity"]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, X.columns.tolist(), metrics, X

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load waste.csv: {exc}")
    st.stop()

if "Waste_Quantity" not in data.columns:
    st.error("The dataset must contain a 'Waste_Quantity' column.")
    st.stop()

if len(data) < 10:
    st.warning("This demo dataset has very few rows. Model scores may not be reliable.")

model, feature_columns, metrics, encoded_data = train_model(data)

left, middle, right = st.columns(3)
left.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.2f}")
middle.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
right.metric("R² score", f"{metrics['r2']:.3f}")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict waste quantity")
st.caption("Enter values for the input features. Units and category options should match your dataset.")

with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, column in enumerate(data.drop(columns=["Waste_Quantity"]).columns):
        series = data[column]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(series):
                values[column] = st.number_input(
                    column,
                    value=float(series.median()),
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                values[column] = st.selectbox(column, options=options)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([values])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    prediction = float(model.predict(input_encoded)[0])
    st.success(f"Estimated waste quantity: **{prediction:.2f}**")
    st.caption("This is a model estimate, not a guaranteed measurement.")

st.subheader("Actual vs predicted (test set)")
X = data.drop(columns=["Waste_Quantity"])
y = data["Waste_Quantity"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
test_predictions = model.predict(X_test)
chart_data = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": pd.Series(test_predictions)
})
st.line_chart(chart_data)

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("rf_dropout_model.joblib")

st.title("🎓 Early Detection of Student Dropout")
st.write("Upload a CSV file with student data to predict dropout risk.")

# File uploader
uploaded_file = st.file_uploader("Upload your student dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # One-hot encode to match training format
    X = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    # Make predictions
    preds = model.predict(X)
    df["Dropout_Prediction"] = preds

    st.subheader("✅ Predictions (0 = continue, 1 = dropout)")
    st.dataframe(df[["Dropout_Prediction"]])

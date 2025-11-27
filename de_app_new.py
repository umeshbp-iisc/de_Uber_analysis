import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# ------------------------------
# STREAMLIT PAGE SETTINGS
# ------------------------------
st.set_page_config(page_title="Uber Ride Prediction Dashboard", layout="wide")
st.title("🚖 Uber Ride Prediction & EDA Dashboard")
st.write("Upload NCR Ride Bookings CSV and run the ML pipeline without Spark.")

# ------------------------------
# FILE UPLOAD
# ------------------------------
data_file = st.file_uploader("Upload NCR Ride Bookings CSV", type=["csv"])

if data_file is not None:
    st.success("File uploaded successfully!")

    df = pd.read_csv(data_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ------------------------------
    # DATA CLEANING
    # ------------------------------
    st.subheader("Data Cleaning")

    numeric_columns = [
        'Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance',
        'Driver Ratings', 'Customer Rating'
    ]

    # Replace string 'null' with NaN
    df.replace("null", np.nan, inplace=True)

    # Convert numeric columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill numeric nulls with 0
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Create timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors='coerce'
    )

    st.write("✔ Data cleaning complete")

    # ------------------------------
    # EDA SECTION
    # ------------------------------
    st.header("📊 Exploratory Data Analysis")

    # 1. Booking status distribution
    st.subheader("1️⃣ Booking Status Distribution")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    df["Booking Status"].value_counts().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

    # 2. Hourly booking trends
    st.subheader("2️⃣ Hourly Booking Trends")
    df["Hour"] = df["Timestamp"].dt.hour
    hourly = df.groupby("Hour")["Booking Status"].value_counts().unstack().fillna(0)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    hourly.plot(kind="bar", stacked=True, ax=ax2)
    st.pyplot(fig2)

    # 3. Correlation heatmap
    st.subheader("3️⃣ Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------
    st.header("🛠 Feature Engineering & ML Model")

    df["RideCompleted"] = df["Booking Status"].apply(lambda x: 1 if x == "Completed" else 0)

    # Label encoding categorical columns
    category_columns = [
        "Vehicle Type", "Pickup Location", "Drop Location", "Payment Method"
    ]

    le_dict = {}
    for col in category_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Feature columns
    feature_cols = numeric_columns + category_columns

    X = df[feature_cols]
    y = df["RideCompleted"]

    # ------------------------------
    # TRAIN-TEST SPLIT
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # TRAIN MODEL
    # ------------------------------
    st.subheader("Training Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ------------------------------
    # EVALUATION
    # ------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.metric("Model Accuracy", f"{accuracy:.4f}")
    st.metric("Model RMSE", f"{rmse:.4f}")

    # ------------------------------
    # DOWNLOAD PREDICTIONS
    # ------------------------------
    st.subheader("Download Predictions")

    pred_df = pd.DataFrame({
        "Prediction": y_pred,
        "Actual": y_test.values
    })

    st.dataframe(pred_df.head())

    csv = pred_df.to_csv(index=False).encode()
    st.download_button("Download Predictions CSV", csv, "predictions.csv")

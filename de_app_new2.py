import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------------------------------------------
# Streamlit Settings
# -------------------------------------------------------
st.set_page_config(page_title="Uber Ride Analytics Dashboard", layout="wide")
st.title("🚖 Uber Ride Analytics, EDA & ML Pipeline")
st.write("Upload NCR Ride Bookings CSV to run analytics")

# -------------------------------------------------------
# Upload CSV
# -------------------------------------------------------
data_file = st.file_uploader("Upload NCR Ride Booking CSV", type=["csv"])

if data_file is None:
    st.info("Please upload CSV to continue.")
    st.stop()

df = pd.read_csv(data_file)

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# -------------------------------------------------------
# CLEANING + FEATURE CREATION
# -------------------------------------------------------
st.header("🧹 Data Cleaning & Feature Engineering")

numeric_columns = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance',
                   'Driver Ratings', 'Customer Rating']

df.replace("null", np.nan, inplace=True)

# Convert to numeric
st.caption("✔ Convert to numeric")
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[numeric_columns] = df[numeric_columns].fillna(0)

# Merge timestamp
st.caption("✔ Merge timestamp")
df["Timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                 errors='coerce')

df["DayOfWeek"] = df["Timestamp"].dt.day_name()
df["HourOfDay"] = df["Timestamp"].dt.hour.fillna(0).astype(int)

# Clean categorical nulls
st.caption("✔ Clean categorical nulls")
cat_defaults = {
    "Reason for cancelling by Customer": "Not cancelled",
    "Driver Cancellation Reason": "Not cancelled",
    "Incomplete Rides Reason": "Not incomplete",
    "Payment Method": "Unknown"
}

for c, v in cat_defaults.items():
    if c in df.columns:
        df[c] = df[c].replace("null", v).fillna(v)

# Cancelled flags
st.caption("✔ Cancelled flags")
df["is_cancelled"] = np.where(
    (df.get("Cancelled Rides by Customer", 0).astype(float) > 0) |
    (df.get("Cancelled Rides by Driver", 0).astype(float) > 0),
    1, 0
)

df["RideCompleted"] = df["Booking Status"].apply(lambda x: 1 if x == "Completed" else 0)

st.success("✔ Data cleaned successfully.")

# -------------------------------------------------------
# SUMMARY METRICS
# -------------------------------------------------------
st.header("📌 Key Metrics")

total_rides = len(df)
completed = df["RideCompleted"].sum()
success_rate = round(100 * completed / total_rides, 2)
acceptance_rate = round(100 * (1 - df["is_cancelled"].mean()), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Rides", total_rides)
col2.metric("Ride Completion Success Rate", f"{success_rate}%")
col3.metric("Ride Acceptance Rate", f"{acceptance_rate}%")


# -------------------------------------------------------
# EDA SECTION
# -------------------------------------------------------
st.header("📊 Exploratory Data Analysis")

sns.set_style("whitegrid")
sns.set_palette("pastel")
df1 = df.copy()

# -------------------------------------------------------
# 1️⃣ Booking Status + Customer Cancellations (SIDE-BY-SIDE)
# -------------------------------------------------------
st.subheader("1️⃣ Booking Status & Hourly Ride Count")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.countplot(data=df, y="Booking Status", ax=ax)
    ax.set_title("Booking Status Distribution")
    st.pyplot(fig)


with col2:
    fig, ax = plt.subplots(figsize=(5,3))
    hourly = df["HourOfDay"].value_counts().sort_index()
    sns.barplot(x=hourly.index, y=hourly.values, ax=ax)
    ax.set_title("Rides by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Ride Count")
    st.pyplot(fig)



# -------------------------------------------------------
# 2️⃣ Driver Cancellations + Hourly Ride Count (SIDE-BY-SIDE)
# -------------------------------------------------------
st.subheader("2️⃣ Driver Cancellations & Customer Cancellations")

col1, col2 = st.columns(2)

with col1:
    if "Cancelled Rides by Driver" in df1.columns:
        fig, ax = plt.subplots(figsize=(4,3))
        values = (
            df1["Cancelled Rides by Driver"]
            .replace(["null", "", " ", None], 0)
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna(0).astype(int)
            .value_counts()
        )
        ax.pie(values, autopct="%1.1f%%", labels=values.index)
        ax.set_title("Driver Cancellations")
        st.pyplot(fig)

with col2:
    if "Cancelled Rides by Customer" in df1.columns:
        fig, ax = plt.subplots(figsize=(4,3))
        values = (
            df1["Cancelled Rides by Customer"]
            .replace(["null", "", " ", None], 0)
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna(0).astype(int)
            .value_counts()
        )
        ax.pie(values, autopct="%1.1f%%", labels=values.index)
        ax.set_title("Customer Cancellations")
        st.pyplot(fig)

# -------------------------------------------------------
# 3️⃣ Booking Value by Hour + Vehicle Types (SIDE-BY-SIDE)
# -------------------------------------------------------
st.subheader("3️⃣ Booking Value & Vehicle Types")

col1, col2 = st.columns(2)

with col1:
    tmp = df.groupby("HourOfDay")["Booking Value"].mean()
    fig, ax = plt.subplots(figsize=(5,3))
    sns.lineplot(x=tmp.index, y=tmp.values, marker="o", ax=ax)
    ax.set_title("Avg Booking Value by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Value")
    st.pyplot(fig)

with col2:
    if "Vehicle Type" in df.columns:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(data=df, y="Vehicle Type", ax=ax)
        ax.set_title("Rides by Vehicle Type")
        st.pyplot(fig)


# -------------------------------------------------------
# 4️⃣ Vehicle Type vs Booking Status  (SIDE-BY-SIDE BAR)
# -------------------------------------------------------
st.subheader("4️⃣ Vehicle Type vs Booking Status")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,3))
    crosstab_df = pd.crosstab(df['Vehicle Type'], df['Booking Status'])
    crosstab_df.plot(kind="bar", ax=ax, legend=True)
    ax.set_title("Vehicle Type vs Booking Status")
    ax.set_ylabel("Count")
    ax.set_xlabel("Vehicle Type")
    st.pyplot(fig)


# -------------------------------------------------------
# 6️⃣ Reasons for Customer Cancellations
# -------------------------------------------------------
with col2:
    if "Reason for cancelling by Customer" in df.columns:
        fig, ax = plt.subplots(figsize=(5,3))
        reason_counts = (
            df["Reason for cancelling by Customer"]
            .value_counts()
            .iloc[1:]       # exclude "Not cancelled"
        )

        sns.barplot(
            x=reason_counts.values,
            y=reason_counts.index,
            ax=ax
        )

        ax.set_title("Reasons for Cancelling")
        ax.set_xlabel("Cancellations")
        ax.set_ylabel("Reason")
        plt.xticks(rotation=30)
        st.pyplot(fig)

st.success("✔ EDA done successfully.")
# -------------------------------------------------------
# 4️⃣ Vehicle vs Customer Cancellation (FULL WIDTH)
# -------------------------------------------------------
# st.subheader("5️⃣ Customer Cancellations by Vehicle Type")

# fig, ax = plt.subplots(figsize=(8,4))
# sns.barplot(data=df, x="Vehicle Type", y="Cancelled Rides by Customer", ax=ax)
# ax.set_title("Cancellation Count by Vehicle Type")
# st.pyplot(fig)

# -------------------------------------------------------
# MACHINE LEARNING PIPELINE (Python version)
# -------------------------------------------------------
st.header("🤖 Machine Learning Model")

# Label encoding
categorical_cols = ["Vehicle Type", "Pickup Location", "Drop Location"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

feature_cols = ["Avg VTAT", "Avg CTAT", "Ride Distance",
                "Vehicle Type", "Pickup Location", "Drop Location"]

X = df[feature_cols]
y = df["RideCompleted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- MODELS --------------------
# -------------------- MODELS --------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

results = {}

import time

for name, model in models.items():
    start = time.time()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    end = time.time()
    runtime = round(end - start, 4)    # runtime in seconds

    acc = accuracy_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results[name] = {
        "Accuracy": acc,
        "RMSE": rmse,
        "Runtime (sec)": runtime
    }

# Display results
st.subheader("📌 Model Comparison (Accuracy, RMSE, Runtime)")
res_df = pd.DataFrame(results).T
st.dataframe(
    res_df.style.format({
        "Accuracy": "{:.4f}",
        "RMSE": "{:.4f}",
        "Runtime (sec)": "{:.4f}"
    })
)


# -------------------------------------------------------
# Download Predictions
# -------------------------------------------------------
best_model_name = res_df["Accuracy"].idxmax()
st.success(f"🏆 Best Model: {best_model_name}")

best_model = models[best_model_name]
pred = best_model.predict(X_test)

out_df = pd.DataFrame({"Prediction": pred, "Actual": y_test.values})
csv = out_df.to_csv(index=False).encode()

st.download_button("Download Predictions CSV", csv, "predictions.csv")

st.success("✔ ML completed successfully.")
# -------------------------------------------------------
# 🔥 H2O AutoML (Optional Advanced ML Section)
# -------------------------------------------------------
st.header("🚀 H2O AutoML (Advanced Model)")

run_h2o = st.button("Run H2O AutoML 🚀")

if run_h2o:

    with st.spinner("Initializing H2O and running AutoML... Please wait ⏳"):

        try:
            import h2o
            from h2o.automl import H2OAutoML

            # Start H2O
            if not h2o.connection():
                h2o.init()

            # Convert Pandas -> H2OFrame
            hf = h2o.H2OFrame(df)

            target = "RideCompleted"
            ignore_cols = ["Booking Status", "Date", "Time"]
            features = [c for c in hf.columns if c not in [target] + ignore_cols]

            # Convert target to categorical
            hf[target] = hf[target].asfactor()

            # Run AutoML
            aml = H2OAutoML(
                max_models=20,
                seed=42,
                nfolds=5
            )
            aml.train(x=features, y=target, training_frame=hf)

            st.success("🎉 H2O AutoML Training Completed")

            # Show leaderboard
            st.subheader("🏆 H2O AutoML Leaderboard")
            lb = aml.leaderboard.as_data_frame()
            st.dataframe(lb)

            # Best model
            best_model = aml.leader
            st.success(f"Best Model: {best_model.model_id}")

            # Download best model
            model_path = h2o.save_model(best_model, path="h2o_best_model", force=True)

            with open(model_path, "rb") as f:
                st.download_button(
                    "⬇ Download Best H2O Model",
                    f,
                    file_name="best_h2o_model.zip"
                )

        except Exception as e:
            st.error(f"⚠ H2O AutoML could not run: {e}")
            st.info("Install H2O using:  \n\n`pip install h2o`")

else:
    st.info("Click the **Run H2O AutoML 🚀** button to start AutoML.")


st.success("✔ Pipeline completed successfully!")

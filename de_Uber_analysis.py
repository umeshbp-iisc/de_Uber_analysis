#import required libraries

from narwhals import lit
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

import matplotlib.pyplot as plt
import pandas as pd

# ---------------- STEP 1: Spark Session with Tuning ----------------
#Reduce shuffle partitions and increase executor memory for better performance
#Allocate 4 GB memory to both driver and executor, set 2 cores for executor
# set default parallelism to 200

spark = SparkSession.builder \
    .appName("UberETLMLPipeline") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "200") \
    .getOrCreate()

# ---------------- STEP 2: Load and Preprocess data----------------

#
data_file = "ncr_ride_bookings.csv"

input_df = spark.read.csv(data_file, header=True)
#input_df.show()

# Merge Date and Time into Timestamp
#convert date and time to right format before merge
input_df = input_df.withColumn("Timestamp", to_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss"))

#Drop original Date and Time columns
#input_df = input_df.drop("Date", "Time")
#input_df.show()

# ---------------- STEP 3: Data Cleaning ----------------

# Replace 'null' strings with actual nulls, then cast to double
#VTAT : Vehicle turn around time
#CTAT : Customer turn around time
numeric_columns = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']

for column in numeric_columns:
    if column in input_df.columns:
        # Replace 'null' string with None, then cast to double
        input_df = input_df.withColumn(column, 
                                       when(col(column) == "null", None)
                                       .otherwise(col(column))
                                       .cast("double"))

#Data cleaning: Fill null with default values 
input_df = input_df.fillna({
    'Avg VTAT': 0.0,
    'Avg CTAT': 0.0,
    'Booking Value': 0.0,
    'Ride Distance': 0.0,
    'Driver Ratings': 0.0,
    'Customer Rating': 0.0
})
input_df.show()

######################################### >EDA<  #########################################
# Step 4 EDA : Visualization of cleaned data

# #1 Convert Spark DataFrame to Pandas DataFrame for visualization
pandas_df = input_df.toPandas()

# plot booking status distribution
plt.figure(figsize=(8, 8))
pandas_df['Booking Status'].value_counts().plot(kind='bar')
plt.title("Booking Status Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Count")
#split x-axis labels to 2 lines for better readability
labels = [label.get_text() for label in plt.gca().get_xticklabels()]
labels = [label.replace(' ', '\n') for label in labels]
plt.xticks(range(len(labels)), labels, rotation=0)
#plt.show()

#2 Time of day trend of bookings
# Create a new column for hour of the day
pandas_df['Hour'] = pandas_df['Timestamp'].dt.hour
# Model data
hourly_bookings = pandas_df.groupby('Hour')['Booking Status'].value_counts().unstack().fillna(0)
hourly_bookings.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Hourly Booking Status Distribution")
plt.xlabel("Hour of the Day")
plt.ylabel("Count")
plt.xticks(rotation=0)
#plt.show()

#3 Correlation heatmap of numerical features
import seaborn as sns
correlation_matrix = pandas_df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
#plt.show()

#4 Cancellation Reason Analysis by Customer

plt.figure(figsize=(10, 5))
sns.countplot(data=pandas_df[pandas_df["Cancelled Rides by Customer"] == '1'],
              x="Reason for cancelling by Customer",
              order=pandas_df["Reason for cancelling by Customer"].value_counts().index)
plt.title("Cancellation Reasons by Customer")
plt.ylabel("Count")
plt.xlabel("Reason for Cancelling by Customer")
#split x-axis labels to 2 lines for better readability
labels = [label.get_text() for label in plt.gca().get_xticklabels()]
labels = [label.replace(' ', '\n') for label in labels]
plt.xticks(range(len(labels)), labels, rotation=0)
plt.show()


#5 Cancellation Reason Analysis by Driver
plt.figure(figsize=(10, 5))
sns.countplot(data=pandas_df[pandas_df["Cancelled Rides by Driver"] == '1'],
              x="Driver Cancellation Reason",
              order=pandas_df["Driver Cancellation Reason"].value_counts().index)
plt.title("Cancellation Reasons by Driver")
plt.ylabel("Count")
plt.xlabel("Reason for Cancelling by Driver")
#split x-axis labels to 2 lines for better readability
labels = [label.get_text() for label in plt.gca().get_xticklabels()]
labels = [label.replace(' ', '\n') for label in labels]
plt.xticks(range(len(labels)), labels, rotation=0)
plt.show()

#6 Cancellation statistics by vehicle type
vehicle_cancellations = pandas_df[pandas_df["Cancelled Rides by Customer"] == '1']['Vehicle Type'].value_counts()
vehicle_cancellations.plot(kind='bar', figsize=(10, 6))

plt.title("Cancellation Statistics by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Count")
#split x-axis labels to 2 lines for better readability
labels = [label.get_text() for label in plt.gca().get_xticklabels()]
labels = [label.replace(' ', '\n') for label in labels]
plt.xticks(range(len(labels)), labels, rotation=45)
plt.show()


# 7. Time-of-Day Booking Trends
# Convert Date and Time to datetime
pandas_df["Timestamp"] = pd.to_datetime(pandas_df["Date"].astype(str) + ' ' + pandas_df["Time"].astype(str), errors='coerce')
pandas_df["Hour"] = pandas_df["Timestamp"].dt.hour

# Define time of day bins
def time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

pandas_df["Time of Day"] = pandas_df["Hour"].apply(time_of_day)
plt.figure(figsize=(8, 5))
sns.countplot(data=pandas_df, x="Time of Day", order=["Morning", "Afternoon", "Evening", "Night"])
plt.title("Time-of-Day Booking Trends")
plt.xlabel("Time of Day")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

############################################################

# #Step 5: Modeling and Evaluation


# # Create target column
# input_df = input_df.withColumn("RideCompleted", when(col("Booking Status") == "Completed", 1).otherwise(0))


# # Dimension tables
# dim_customer = input_df.select("Customer ID", "Customer Rating").dropDuplicates()
# dim_customer.show(5)
# dim_driver = input_df.select("Driver Ratings").dropDuplicates()
# dim_driver.show(5)
# dim_location = input_df.select("Pickup Location", "Drop Location").dropDuplicates()
# dim_location.show(5)
# dim_vehicle = input_df.select("Vehicle Type").dropDuplicates()
# dim_vehicle.show(5)

# # Fact table
# fact_rides = input_df.select("Booking ID", "Customer ID", "Vehicle Type", "Pickup Location", "Drop Location",
#                              "Avg VTAT", "Avg CTAT", "Booking Value", "Ride Distance", "Driver Ratings",
#                              "Customer Rating", "Payment Method", "Timestamp", "RideCompleted")

# fact_rides.show(5)

# #step 6: feature engineering
# # String Indexing for categorical features
# indexers = [
#     StringIndexer(inputCol="Vehicle Type", outputCol="VehicleType_Index"),
#     StringIndexer(inputCol="Pickup Location", outputCol="PickupLocation_Index"),
#     StringIndexer(inputCol="Drop Location", outputCol="DropLocation_Index"),
#     StringIndexer(inputCol="Payment Method", outputCol="PaymentMethod_Index")
# ]

# # Apply indexers
# for indexer in indexers:
#     fact_rides = indexer.fit(fact_rides).transform(fact_rides)

# # Define feature columns
# feature_cols = ["Avg VTAT", "Avg CTAT", "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating",
#                 "VehicleType_Index", "PickupLocation_Index", "DropLocation_Index", "PaymentMethod_Index"]

# # Assemble features into a single vector
# # Vector Assembler is selected here to combine multiple feature columns into a single feature vector for model training.
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# # Transform fact_rides to include features column and select features and target column
# df_model = assembler.transform(fact_rides).select("features", "RideCompleted")

# # Step 7: Split data into training and testing sets
# train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

# # step 8: Model Training
# #Random Forest Classifier is selected here for its robustness and ability to handle both numerical and categorical features effectively.
# # Increase maxBins to handle categorical features with many unique values
# rf_classifier = RandomForestClassifier(labelCol="RideCompleted", featuresCol="features", numTrees=50, maxDepth=10, maxBins=200, seed=42)
# # Fit the model on training data 
# rf_model = rf_classifier.fit(train_df)

# # Make predictions on test data
# predictions = rf_model.transform(test_df)

# # Step 9: Model Evaluation
# evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="RideCompleted", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator_accuracy.evaluate(predictions)

# #step 10  rmse evaluator for regression
# evaluator_rmse = RegressionEvaluator(labelCol="RideCompleted", predictionCol="prediction", metricName="rmse")
# rmse = evaluator_rmse.evaluate(predictions)

# print(f"Model Accuracy: {accuracy:.4f}")
# print(f"Model RMSE: {rmse:.4f}")

# #step 11: Save outputs
# predictions.select("prediction", "RideCompleted").write.mode("overwrite").option("header", True).csv("uber_ride_predictions.csv")



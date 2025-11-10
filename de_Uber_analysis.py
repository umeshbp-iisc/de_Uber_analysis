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

# Visualization of cleaned data

#1 Convert Spark DataFrame to Pandas DataFrame for visualization
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
plt.show()

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
plt.show()

#3 Correlation heatmap of numerical features
import seaborn as sns
correlation_matrix = pandas_df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.show()

#Cancellation Reason Analysis

plt.figure(figsize=(10, 5))
sns.countplot(data=pandas_df[pandas_df["Cancelled Rides by Customer"] == 1],
              y="Reason for cancelling by Customer",
              order=pandas_df["Reason for cancelling by Customer"].value_counts().index)
plt.title("Cancellation Reasons by Customer")
plt.show()


#Cancellation statistics by vehicle type
# vehicle_cancellations = pandas_df[pandas_df["Cancelled Rides by Customer"] == 1]['Vehicle Type'].value_counts()
# vehicle_cancellations.plot(kind='bar', figsize=(10, 6))

# plt.title("Cancellation Statistics by Vehicle Type")
# plt.xlabel("Vehicle Type")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()

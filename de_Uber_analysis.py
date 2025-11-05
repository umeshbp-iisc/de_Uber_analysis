#import required libraries

from narwhals import lit
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator


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
input_df = input_df.drop("Date", "Time")
#input_df.show()

# ---------------- STEP 3: Data Cleaning ----------------

# Replace 'null' strings with actual nulls, then cast to double
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




#import data



#clean data





#model data



#analyze data




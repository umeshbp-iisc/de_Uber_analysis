#import pyspark libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, hour 
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
# Create Spark session
spark = SparkSession.builder \
    .appName("Uber Data Analysis") \
    .getOrCreate()  


#import data



#clean data





#model data



#analyze data




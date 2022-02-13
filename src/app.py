import sys, math
from os import path, getenv

import boto3
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window




LOCAL_BASE_PATH = getenv('BASE_PATH')
REMOTE_PATH = getenv('REMOTE_DATA_PATH')
BUCKET_STORAGE = getenv('BUCKET_STORAGE')


try:
    sc = SparkContext('local', 'Pyspark demo')
except ValueError:
    print('SparkContext already exists!')

try:
    spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()
except ValueError:
    print('SparkSession already exists!')


print('Downloading file')
LOCAL_TRAINED_PATH = path.join(LOCAL_BASE_PATH,
                                  'data_trained_for_model.parquet')
LOCAL_PATH = path.join(LOCAL_BASE_PATH,
                          'dataset_credit_risk.csv')
s3 = boto3.resource('s3')
s3.Object(BUCKET_STORAGE, REMOTE_PATH).download_file(LOCAL_PATH)
print('File downloaded')

with open(LOCAL_PATH) as f:
    data = f.readlines()
data.__len__()


n_slices = math.ceil(sys.getsizeof(data) / 81920)
n_slices
print('Creating spark rdd')
spark_df = spark.read.csv(sc.parallelize(data, numSlices=n_slices), header=True)

spark_df = spark_df.drop(
 'loan_id',
 'code_gender',
 'flag_own_realty',
 'cnt_children',
 'amt_income_total',
 'name_income_type',
 'name_education_type',
 'name_family_status',
 'name_housing_type',
 'days_birth',
 'days_employed',
 'flag_mobil',
 'flag_work_phone',
 'flag_phone',
 'flag_email',
 'occupation_type',
 'cnt_fam_members',
 'status')

print('Generating feature nb_previous_loans')
# Feature nb_previous_loans
window_spec  = Window.partitionBy("id").orderBy("loan_date")
spark_df = spark_df.withColumn("nb_previous_loans",F.row_number().over(window_spec))
spark_df = spark_df.withColumn('nb_previous_loans', F.col('nb_previous_loans')-1)

print('Generating feature avg_amount_loans_previous')

# Feature avg_amount_loans_previous
window = Window.orderBy("id").partitionBy("id").rowsBetween(Window.unboundedPreceding, -1)
spark_df = spark_df.withColumn("avg_amount_loans_previous", F.mean('loan_amount').over(window))

print('Generating feature age')

# Feature age
spark_df = spark_df.withColumn("age",F.floor(F.datediff(F.current_date(),F.col("birthday"))/F.lit(365)))

print('Generating feature year_on_the_job')

#Feature year_on_the_job
spark_df = spark_df.withColumn("years_on_the_job",F.floor(F.datediff(F.current_date(),F.col("job_start_date"))/F.lit(365)))

print('Generating feature flag_own_car')

# Feature flag_own_car
spark_df = spark_df.withColumn('flag_own_car', F.when(spark_df['flag_own_car'] == 'N', 0).otherwise(1))

print('Saving trained results')

# convert to pandas dataframe
df_processed = spark_df[['id', 
                        'age', 
                        'years_on_the_job', 
                        'nb_previous_loans', 
                        'avg_amount_loans_previous', 
                        'flag_own_car']].toPandas()
# df_processed.to_csv(LOCAL_TRAINED_PATH)
df_processed.to_parquet(LOCAL_TRAINED_PATH)
REMOTE_TRAINED_PATH = 'processed_data/trained_model.parquet'
# store on s3
s3.meta.client.upload_file(LOCAL_TRAINED_PATH, BUCKET_STORAGE, REMOTE_TRAINED_PATH)
import pyspark
import findspark
import os
from pyspark.sql.functions import udf
import pickle

from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType

if __name__ == '__main__':
    print(findspark.find())
    findspark.init()

    sc = pyspark.SparkContext()

    scala_version = '2.12'
    spark_version = sc.version

    # os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:{}'.format(sc.version)

    packages = [
        f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
        'org.apache.kafka:kafka-clients:2.6.0'
    ]

    spark = pyspark.sql.SparkSession \
        .builder \
        .master('local') \
        .appName("test") \
        .config("spark.jars.packages", ",".join(packages)) \
        .getOrCreate()

    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("startingOffsets", "earliest") \
        .option("subscribe", "twitter-events") \
        .load()

    tf_idf = open("BDA/src/model/cv.pkl", "rb")
    cv = pickle.load(tf_idf)

    # import pickle file of my model
    model = open("BDA/src/model/model.pkl", "rb")
    clf = pickle.load(model)

    @udf
    def mapping(x):
        y = clf.predict(cv.transform([x]).toarray())[0]
        return int(y)

    schema = StructType([
        StructField("text", StringType(), True),
        ])

    (kafka_df
        # .withColumnRenamed('value', 'text')
        # .selectExpr("CAST(text AS STRING)")
        .selectExpr("CAST(value AS STRING)", "timestamp")
        .select(from_json('value', schema).alias("parsed_value"), 'timestamp')
        .select("parsed_value.text", 'timestamp')
        .withColumn('evaluation', mapping('text'))

        .writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", "BDA/hdfs/batch_view")
        # .format('json')
        # .option("path", "BDA/src/log")
        .option("header", True)
        .option("checkpointLocation", "BDA/hdfs/checkpoint")
        .start()
        .awaitTermination())

    print('--- DONE')



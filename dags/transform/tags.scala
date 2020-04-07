val tags_df = spark.read.format("csv").option("header", "true").option("inferschema", "true").load("s3://<s3-bucket>/tags.csv")

tags_df.write.mode("overwrite").parquet("s3://<s3-bucket>/movielens-parquet/tags/")

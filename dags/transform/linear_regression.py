from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import mlflow
import mlflow.spark

completa_metricas = spark.read.parquet("s3://<s3-bucket>/movielens-parquet/training/")

columnas_input = list(completa_metricas.columns)
columnas_input.remove('promedio_rating')

vectorAssembler = VectorAssembler(inputCols = columnas_input, outputCol = 'features')
completa_metricas_vector = vectorAssembler.transform(completa_metricas)

splits = completa_metricas_vector.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]

mlflow.set_tracking_uri("http://cambiame:5000")
maxIter = 100
elasticNetParam = 0.1
regParam=0.1
                        
lr = LinearRegression(featuresCol='features',labelCol='promedio_rating',maxIter=maxIter,elasticNetParam=elasticNetParam,regParam=regParam)

with mlflow.start_run():
    mlflow.log_param("maxIter", maxIter)
    mlflow.log_param("elasticNetParam", elasticNetParam)
    mlflow.log_param("regParam", regParam)

    lr_model = lr.fit(train_df)
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    mlflow.spark.log_model(lr_model, "spark-model")
    mlflow.log_metric("rmse", trainingSummary.rootMeanSquaredError)
    mlflow.log_metric("r2", trainingSummary.r2)
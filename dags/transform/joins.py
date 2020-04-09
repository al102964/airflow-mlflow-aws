import pyspark.sql.functions as f

categorias = ['action','adaptation','adventure','apocalypse','artistic',\
              'assassination','based on a true story','biblical','blood','brutal','biographical','bollywood','boring',\
              'cars','cerebral','classic','censorship','comedy','computers','confusing',\
              'cooking','comic','cartoon','court','crime','cult','dark','death','disaster','documentary','drama',
              'depressing','drugs','environment','erotic','fantasy','fighting','football','freedom',\
              'friendship','genius','god','gothic','high school','historical','hollywood','horror',\
              'humor','homosexuality','holiday','independent film','kids','love','magic','marriage',\
              'military','murder','musical','nature','nostalgia','nudity','olympics','original','oscar',\
              'pirates','police','pornography','prison','prostitution','psychology','realistic',\
              'revolution','robot','romance','scary','science','sex','snakes','soccer','space','sports',\
              'spy','story','stunning','superhero','surreal','suspense','technology','teen','thriller',\
              'time','torture','tragedy','travel','treasure','true story','utopia','3d','war','wizards',\
              'zombie','dreamworks','disney','gore','imdb top 250','literary adaptation','mafia','pixar','sci fi','violence'
             ]

#genome_tags_path = "../../ml-latest/genome-tags.csv"
#ratings_path = "../../ml-latest/ratings.csv"
#genome_scores_path = "../../ml-latest/genome-scores.csv"

genome_tags_path = "s3://<s3-bucket>/movielens-parquet/genome-tags/"
ratings_path = "s3://<s3-bucket>/movielens-parquet/ratings/"
genome_scores_path = "s3://<s3-bucket>/movielens-parquet/genome-scores/"


#genome_tags = spark.read.csv(genome_tags_path,header='true')
#ratings = spark.read.csv(ratings_path,header='true')
#genome_scores = spark.read.csv(genome_scores_path,header='true')

genome_tags = spark.read.parquet(genome_tags_path)
ratings = spark.read.parquet(ratings_path)
genome_scores = spark.read.parquet(genome_scores_path)


genome_tags_filtrado = genome_tags.where(f.col("tag").isin(categorias))
etiquetas_df = genome_scores.join(genome_tags_filtrado,['tagId'],'leftsemi')

etiquetas_relevancia = etiquetas_df.join(genome_tags_filtrado,['tagId'],'left')
etiquetas_relevancia = etiquetas_relevancia.withColumn("tags", f.regexp_replace(f.col("tag"), " ", "_"))
etiquetas_relevancia = etiquetas_relevancia.drop("tag").withColumnRenamed("tags","tag")

completa = etiquetas_relevancia.groupby("movieId").pivot("tag").agg(f.avg('relevance'))

metricas_ratings = ratings.groupby("movieId").agg(f.avg(f.col("rating")).alias("promedio_rating"),
                                                  f.count(f.col("userId")).alias("conteo_usuarios"))

completa_metricas = completa.join(metricas_ratings,['movieId'], 'left').drop("movieId")
completa_metricas.write.mode("overwrite").parquet("s3://<s3-bucket>/movielens-parquet/training/")


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from konlpy.tag import Okt
from pyspark.ml.feature import CountVectorizer

if __name__== "__main__":
	spark = SparkSession.builder.appName("review").getOrCreate()
	df = spark.read.load("BDP_TeamProject/review_subset.csv",format="csv",sep=",",inferSchema="true",header="true")

	df = df.withColumn("year", F.year("at")).withColumn("month", F.month("at"))

	def extract_nouns(text):
		okt = Okt()
		pos_tags = okt.pos(text)
		nouns = [word for word, pos in pos_tags if pos in ["Noun"]]
		#verbs = [f"{prev_word}_{word}" for prev_word, (word, pos) in zip([''] + pos_tags[:-1], pos_tags) if pos in ["Verb"]]
		return nouns

	extract_nouns_udf = F.udf(extract_nouns, ArrayType(StringType()))

	df = df.withColumn("nouns", extract_nouns_udf("content"))
	all_combined_nouns = df.select(F.explode("nouns").alias("word")) \
			.groupBy()\
			.agg(F.collect_list("word").alias("all_combined_nouns"))

	df.show()
	#df.show()
	#df = df.withColumn("verbs", F.col("nouns_verbs").getField("_2"))

	#noun_count = df.select("year", "month", F.explode("nouns").alias("word")).groupBy("year", "month", "word").count()
	#noun_count.show()


	#df.show()
	#word_count = df.select("year", "month", F.explode(F.split("content", " ")).alias("word"))\
	#		.groupBy("year", "month", "word").count()\
	#		.filter(F.length("word")>1)
	
	#word_count.show()
	#aggregated_word_count = word_count.groupBy("year", "month", "word").agg(F.sum("count").alias("total_count"))

	#aggregated_word_count.show()



#	distinct_years = [row['year'] for row in df.select('year').distinct().collect()]
#	distinct_months = [row['month'] for row in df.select('month').distinct().collect()]
	
#	for year in distinct_years:
#		for month in distinct_months:
#			df_filtered = df.filter((df["year"] == year) & (df["month"] == month))
#			output_path = f"BDP_TeamProject/timeData/{year}_{month}"
#			df_filtered.write.csv(output_path, header=True, mode="overwrite")
	
	

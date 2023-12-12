from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from konlpy.tag import Okt
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

if __name__== "__main__":
	spark = SparkSession.builder.appName("review").getOrCreate()
	df = spark.read.load("BDP_TeamProject/review.csv",format="csv",sep=",",inferSchema="true",header="true")
	#stopwords = spark.read.text("BDP_TeamProject/bull.txt")
	

	df = df.na.drop(subset=['content'])
	#df = df.withColumn("year", F.year("at")).withColumn("month", F.month("at"))
	#df = df.select("year","month","score","content")
	print(stopwords)
	def extract_nouns(text):
		okt = Okt()
		pos_tags = okt.pos(text)
		nouns = [word for word, pos in pos_tags\
				if pos in ["Noun"] and len(word) > 1]
		nonus = [x for x in nouns if x not in stopwords]
		result = ' '.join(nouns)
		return result

	extract_nouns_udf = F.udf(extract_nouns,StringType())
	df = df.withColumn("filtered_text", extract_nouns_udf(df["content"]))

	def score_to_label(score):
		return F.when(score > 3, 1).otherwise(0)
	df = df.withColumn('label',score_to_label(df['score']))
	df_sel = df.select("at","score","label","filtered_text")
	df_sel.show()
	
	data = df_sel.select("label", "filtered_text").toPandas()
	train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


	
	#logisticregrssion
	#train_df, test_df = df_sel.randomSplit([0.8, 0.2], seed=42)
	#cv = CountVectorizer(inputCol="filtered_text", outputCol="words")
	#cv_model = cv.fit(df_sel)
	#train_df = cv_model.transform(train_df)
	#test_df = cv_model.transform(test_df)
	

	#lr = LogisticRegression(featuresCol="features", labelCol="label")
	#lr_model = lr.fit(train_df)
	#pred = lr_model.transform(test_df)
	
	#evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
	#auc = evaluator.evaluate(predictions)
	#print(f"Area Under ROC: {auc}")


	#vectorizer = CountVectorizer(inputCol="nouns", outputCol="features", minDF=1)
	#model = vectorizer.fit(sel_df)
	#result = model.transform(sel_df)
	
	#sel_df.show()

	#vocab = model.vocabulary
	#word_count = result.select("features").collect()

	#word_count_dict = {}
	#for row in word_count:
	#	features = row.features.toArray()
	#	for i in range(len(features)):
	#		if features[i] != 0.0:
	#			word_count_dict[vocab[i]] = word_count_dict.get(vocab[i], 0) + int(features[i])
	
	
	#word_count_df = spark.createDataFrame(list(word_count_dict.items()), ["word", "count"])
	#word_count_df = word_count_df.orderBy("count", ascending=False)
	#word_count_df.show(truncate=False)	

	#df_set = df_set.withColumn('nouns', F.concat_ws(' ', 'nouns'))
	#df_set.write.csv("BDP_TeamProject/spark_review", header=True, mode="overwrite")


from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

spark = SparkSession.builder \
		    .appName("ReviewAnalysis") \
			    .master("local[*]") \
				    .getOrCreate()

df = spark.read.load("BDP_TeamProject/review.csv",format="csv",sep=",",inferSchema="true",header="true")

df = df.withColumn("score", df["score"].cast("double"))
df = df.withColumn("thumbsUpCount", df["thumbsUpCount"].cast("double"))
df = df.dropna()

content_list = df.select("content").rdd.flatMap(lambda x: x).collect()

content_list = content_list[:20]
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
tokenized = tokenizer(content_list, padding=True, truncation=True, return_tensors="pt")

dataset = TensorDataset(tokenized['input_ids'], tokenized['attention_mask'])

dataloader = DataLoader(dataset, batch_size=4)

model = BertForSequenceClassification.from_pretrained("monologg/kobert")
model.eval()
predictions = []

with torch.no_grad():
	for batch in dataloader:
		print("Processing batch...")
		outputs = model(batch[0], attention_mask=batch[1])
		predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())


predictions_df = pd.DataFrame(predictions, columns=["predicted_label"])

predictions_col = spark.createDataFrame(predictions_df)

df = df.join(predictions_col, how="outer")

@udf(DoubleType())
def extract_float(value):
	try:
		return float(value)
	except (TypeError, ValueError):
		return None

df = df.withColumn("predicted_label", extract_float(col("predicted_label")))
df = df.withColumn("sentiment", when(col("predicted_label") == 1, "Positive").otherwise("Negative"))

# Function to get top N words and their counts from a list of tokenized words
def get_top_words(words, n=10):
	word_counts = {}
	for word in words:
		if word in word_counts:
			word_counts[word] += 1
		else:
			word_counts[word] = 1
	sorted_words = sorted(word_counts.items(),key=lambda x: x[1], reverse=True)
	return sorted_words[:n]

positive_reviews = df.filter(col("sentiment") == "Positive").select("content").rdd.flatMap(lambda x: x).collect()
negative_reviews = df.filter(col("sentiment") == "Negative").select("content").rdd.flatMap(lambda x: x).collect()

top_positive_words = get_top_words(positive_reviews)
top_negative_words = get_top_words(negative_reviews)

print("Top Positive Words:")
for word, count in top_positive_words:
	    print(f"{word}: {count} times")

print("Top Negative Words:")
for words, count in top_negative_words:
	    print(f"{words}: {count} times")

evaluator = MulticlassClassificationEvaluator(predictionCol="predicted_label", labelCol="score", metricName="accuracy")
accuracy = evaluator.evaluate(df)
print(f"\nAccuracy: {accuracy}")


















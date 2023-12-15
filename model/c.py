import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("learn").getOrCreate()
# 데이터 불러오기
hdfs_path = "hdfs:///user/maria_dev/BDP_TeamProject/kakao.csv/"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
df = df.na.drop(subset=["filtered_nouns_str", "label"])
# Feature Extraction
tokenizer = Tokenizer(inputCol="filtered_nouns_str", outputCol="words")
# Convert array of strings to vector using CountVectorizer
vectorizer = CountVectorizer(inputCol="words", outputCol="features")
idf = IDF(inputCol="features", outputCol="ifd_features")
# Target variable
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
# Train-test split
train, test = df.randomSplit([0.8, 0.2], seed=42)
# Logistic Regression 모델 정의
lr = LogisticRegression(featuresCol="features", labelCol="indexedLabel")
# Pipeline 정의
pipeline = Pipeline(stages=[tokenizer,vectorizer, idf,labelIndexer, lr])
# 모델 학습
model = pipeline.fit(train)
# 예측
predictions = model.transform(test)
# 정확도 계산
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


#logistic Regression model from the pipeline
lr_model = model.stages[-1]

# Get the vocabulary from TF-IDF
vocabulary = model.stages[1].vocabulary

# Get the coefficients from the Logistic Regression model
coefficients = lr_model.coefficients.toArray()

# Create a DataFrame to map feature indices to terms and coefficients
coefficients_df = pd.DataFrame({'term': vocabulary, 'coefficient': coefficients})

# Sort the DataFrame by coefficient values
coefficients_df = coefficients_df.sort_values(by='coefficient', ascending=False)

# Print the top 10 words with the largest coefficients
print("Top 10 words with the largest coefficients:")
print(coefficients_df.head(10))

# Print the top 10 words with the smallest coefficients
print("\nTop 10 words with the smallest coefficients:")
print(coefficients_df.tail(10))

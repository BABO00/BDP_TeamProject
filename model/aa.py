from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

hdfs_path = "hdfs:///user/maria_dev/BDP_TeamProject/kakao.csv/"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True) 
df = df.na.drop(subset=["filtered_nouns_str", "label"])

# Feature Extraction
tokenizer = Tokenizer(inputCol="filtered_nouns_str", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Target variable
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")

# Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="indexedLabel", regParam=0.0)

# Pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, labelIndexer, lr])

# Train-test split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Model training
model = pipeline.fit(train)

# Predictions
predictions = model.transform(test)

# Accuracy calculation
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Logistic Regression model from the pipeline
lr_model = model.stages[-1]

# Get the vocabulary from TF-IDF
vocabulary = model.stages[2].vocabulary

# Get the coefficients from the Logistic Regression model
coefficients = lr_model.coefficients.toArray()

# Create a DataFrame to map feature indices to terms and coefficients
coefficients_df = spark.createDataFrame(list(zip(vocabulary, coefficients)), ["term", "coefficient"])

# Sort the DataFrame by coefficient values
coefficients_df = coefficients_df.orderBy("coefficient", ascending=False)

# Print the top 10 words with the largest coefficients
print("Top 10 words with the largest coefficients:")
coefficients_df.show(10, truncate=False)

# Print the top 10 words with the smallest coefficients
print("\nTop 10 words with the smallest coefficients:")
coefficients_df.orderBy("coefficient").show(10, truncate=False)


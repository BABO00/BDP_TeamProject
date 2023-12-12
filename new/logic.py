from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType
from konlpy.tag import Okt
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Spark 세션 생성
spark = SparkSession.builder \
	.appName("KoreanTextProcessing") \
	.config("spark.executor.memory", "2g") \
	.config("spark.executor.memoryOverhead", "1g") \
	.config("spark.executor.instances", "4") \
	.getOrCreate()

# 데이터 불러오기
data_path = "hdfs:///user/maria_dev/new/kakao.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Konlpy를 이용하여 명사 형태소 추출하는 함수 정의
def extract_nouns(text):
	if text;
	okt = Okt()
	nouns = okt.nouns(text)
	return nouns
	else:
		return []

# PySpark UDF (User Defined Function) 등록
extract_nouns_udf = udf(extract_nouns, ArrayType(StringType()))

# 명사 형태소 추출
df = df.withColumn("nouns", extract_nouns_udf(df["content"]))

# HDFS에 저장된 불용어 리스트 파일 경로
stopwords_hdfs_path = "hdfs:///user/maria_dev/new/bull.txt"

# 불용어를 리스트로 로드
stopwords_list = spark.read.text(stopwords_hdfs_path).rdd.flatMap(lambda x: x).collect()

# StopWordsRemover를 이용하여 불용어 제거
stopwords_remover = StopWordsRemover(inputCol="nouns", outputCol="filtered_nouns", stopWords=stopwords_list)
df = stopwords_remover.transform(df)

# score에 따라 레이블 생성
df = df.withColumn("label", (df["score"] >= 4).cast(IntegerType()))

# 필요한 컬럼 선택
result_df = df.select("score", "filtered_nouns", "label")

# Spark 세션 종료
spark.stop()

# Sklearn을 사용하여 데이터 전처리 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(result_df["filtered_nouns"], result_df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = SklearnLR()
classifier.fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("전체 데이터에 대한 정확도:", accuracy)

# 상위 10개의 긍정 단어와 부정 단어 출력
coef = classifier.coef_.flatten()
positive_top_10 = [vectorizer.get_feature_names()[i] for i in coef.argsort()[-10:][::-1]]
negative_top_10 = [vectorizer.get_feature_names()[i] for i in coef.argsort()[:10]]

print("Top 10 Positive Words:", positive_top_10)
print("Top 10 Negative Words:", negative_top_10)

# WordCloud 생성 및 출력
positive_words = " ".join(positive_top_10)
negative_words = " ".join(negative_top_10)

positive_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_words)
negative_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_words)

# 긍정 단어 WordCloud 출력
plt.figure(figsize=(10, 5))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Top 10 Positive Words')
plt.axis('off')
plt.show()

# 부정 단어 WordCloud 출력
plt.figure(figsize=(10, 5))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Top 10 Negative Words')
plt.axis('off')
plt.show()

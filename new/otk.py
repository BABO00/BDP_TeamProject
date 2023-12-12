from pyspark.sql.functions import concat_ws
from pyspark.sql import SparkSession
from konlpy.tag import Okt
from pyspark.sql.functions import udf, explode, col
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.ml.feature import StopWordsRemover

# Spark 세션 생성
#spark = SparkSession.builder.appName("KoreanTextProcessing").getOrCreate()
spark = SparkSession.builder \
	.appName("KoreanTextProcessing") \
	.config("spark.executor.memory", "2g") \
	.config("spark.executor.memoryOverhead", "1g") \
	.config("spark.executor.instances", "4") \
	.getOrCreate()

# 데이터 불러오기
data_path = "hdfs:///user/maria_dev/new/review.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Konlpy를 이용하여 명사 형태소 추출하는 함수 정의
def extract_nouns(text):
	okt = Okt()
	nouns = okt.nouns(text)
	return nouns

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
result_df = df.select("score","filtered_nouns", "label")

# 결과 출력
result_df.show(truncate=False)

# Spark 세션 종료
#spark.stop()

# Vectorization
# Vectorization
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# 띄어쓰기로 구분된 문자열로 단어 목록을 결합하는 UDF 정의
join_udf = udf(lambda x: ' '.join(x), StringType())

# UDF를 적용하여 새로운 열 생성
result_df = result_df.withColumn("joined_nouns", join_udf(result_df["filtered_nouns"]))

# PySpark DataFrame을 Pandas DataFrame으로 변환
pandas_df = result_df.select("label", "joined_nouns").toPandas()

# 이제 Pandas DataFrame에서 scikit-learn을 사용할 수 있습니다.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(result_df['filtered_nouns'].apply(lambda x: ' '.join(x)))
y = result_df['label']

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")





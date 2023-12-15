from pyspark.sql.functions import concat_ws
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from konlpy.tag import Okt
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import ArrayType, StringType, IntegerType

# Spark 세션 생성
spark = SparkSession.builder \
	.appName("KoreanTextProcessing") \
	.config("spark.executor.memory", "2g") \
	.config("spark.executor.memoryOverhead", "1g") \
	.config("spark.executor.instances", "4") \
	.getOrCreate()

# 데이터 불러오기
data_path = "hdfs:///user/maria_dev/BDP_TeamProject/review.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Konlpy를 이용하여 명사 형태소 추출하는 함수 정의
def extract_nouns(text):
	if text:
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
stopwords_hdfs_path = "hdfs:///user/maria_dev/BDP_TeamProject/bull.txt"

# 불용어를 리스트로 로드
stopwords_list = spark.read.text(stopwords_hdfs_path).rdd.flatMap(lambda x: x).collect()

# StopWordsRemover를 이용하여 불용어 제거
stopwords_remover = StopWordsRemover(inputCol="nouns", outputCol="filtered_nouns", stopWords=stopwords_list)
df = stopwords_remover.transform(df)
# score에 따라 레이블 생성
df = df.withColumn("label", (df["score"] >= 4).cast(IntegerType()))

# 배열 열을 문자열로 변환
df = df.withColumn("filtered_nouns_str", concat_ws(" ", "filtered_nouns"))

# 필요한 열 선택
result_df = df.select("score", "filtered_nouns_str", "label")

# 결과를 CSV 파일로 저장
output_path = "hdfs:///user/maria_dev/BDP_TeamProject/kakao.csv"
result_df.write.csv(output_path, header=True, mode="overwrite")

# 결과 출력
result_df.show(truncate=False)

# Spark 세션 종료
spark.stop()


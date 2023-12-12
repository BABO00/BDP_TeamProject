# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, monotonically_increasing_id, expr
from pyspark.sql.types import StringType
import re

# 1. Spark 세션 생성 (Create Spark Session)
spark = SparkSession.builder.appName("KakaoTalkSentimentAnalysis").getOrCreate()

# 2. CSV 파일 읽기 (Read CSV File)
file_path = "/user/maria_dev/kakaotalk_review.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 3. 전처리 함수 정의 (Define Preprocessing Function)
def extract_word(text):
    hangul = re.compile('[^가-힣]')
    result = hangul.sub(' ', text)
    return result

# 4. UDF(사용자 정의 함수) 등록 
extract_word_udf = udf(extract_word, StringType())

# 5. 필요한 컬럼 선택 (Select Necessary Columns)
df = df.select("content", "thumbsUpCount", "score")

# 7. Null 값 제거 (Remove Null Values)
df = df.dropna()

# 8. 중복 행 제거 (Remove Duplicate Rows)
df = df.dropDuplicates()

# 9. 인덱스 재배열 (Reorder Index)
df = df.withColumn("index", monotonically_increasing_id())

# 전처리 (Register UDF and Preprocessing)
df = df.withColumn("cleaned_content", extract_word_udf(df['content']))

# 10. content의 앞 30글자만 추출하여 shortened_content 컬럼 추가
df = df.withColumn("shortened_content", expr("substring(cleaned_content, 1, 30)"))

# 11. 필요한 컬럼 선택 (Select Necessary Columns)
df = df.select("shortened_content", "thumbsUpCount", "score")

# 12. 전처리 완료된 DataFrame 출력 (Show Processed DataFrame)
df.show(10, truncate=False)

# 13. Spark 세션 종료 (Stop Spark Session)
spark.stop()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 불러오기
data_path1 = "/home/maria_dev/BDP_TeamProject/model/kakao.csv/part-00000-6783050b-41e0-440f-9efc-cb71efae9ea7-c000.csv"
df1 = pd.read_csv(data_path1, engine='python')

data_path2 = "/home/maria_dev/BDP_TeamProject/model/kakao.csv/part-00001-6783050b-41e0-440f-9efc-cb71efae9ea7-c000.csv"
df2 = pd.read_csv(data_path2, engine='python')

# Adding columns for negative_word_count and positive_word_count
df1['negative_word_count'] = 0
df1['positive_word_count'] = 0

df2['negative_word_count'] = 0
df2['positive_word_count'] = 0

df = pd.concat([df1, df2], ignore_index=True)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["filtered_nouns_str"].astype('str'))

# Target variable
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 긍정단어와 부정단어 확인
feature_names = vectorizer.get_feature_names()
coefficients = model.coef_.flatten()

# 모든 긍정단어와 부정단어
all_positive_words = [feature_names[i] for i in coefficients.argsort() if coefficients[i] > 0]
all_negative_words = [feature_names[i] for i in coefficients.argsort() if coefficients[i] < 0]

# 각 리뷰에 대한 긍정단어와 부정단어의 개수 파악
def count_positive_words(x):
	if isinstance(x, str):
		return sum(1 for word in x.split() if word in all_positive_words)
	else:
		return 0
def count_negative_words(x):
	if isinstance(x, str):
		return sum(1 for word in x.split() if word in all_negative_words)
	else:
		return 0

print(f"Original Average Rating: {df['score'].mean()}")
df['positive_word_count'] = df['filtered_nouns_str'].apply(count_positive_words)
df['negative_word_count'] = df['filtered_nouns_str'].apply(count_negative_words)

# 데이터 확인
print("False Positive")
print(df[(df['score'].isin([4, 5])) & (df['positive_word_count'] < df['negative_word_count'])]["filtered_nouns_str"].head(2))
print("False Negetive")
print(df[(df['score'].isin([1, 2, 3])) & (df['positive_word_count'] > df['negative_word_count'])]["filtered_nouns_str"].head(2))

# 4,5점에 부정단어가 더 많거나 1,2,3점에 긍정단어가 더 많은 리뷰 삭제
df = df[~(((df['score'] == 4) | (df['score'] == 5)) & (df['positive_word_count'] < df['negative_word_count']))]
df = df[~(((df['score'] == 1) | (df['score'] == 2) | (df['score'] == 3)) & (df['positive_word_count'] > df['negative_word_count']))]

# 중복 코드를 제거하고 평균평점
print(f"Average Rating after Removing False Review: {df['score'].mean()}")




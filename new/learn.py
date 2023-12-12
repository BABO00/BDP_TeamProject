import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 불러오기
data_path1 = "/home/maria_dev/BDP_TeamProject/new/kakao.csv/part-00001-0ab6da8b-c151-4159-ad13-4f3b9804cceb-c000.csv"
df1 = pd.read_csv(data_path1, engine='python')

data_path2 = "/home/maria_dev/BDP_TeamProject/new/kakao.csv/part-00000-0ab6da8b-c151-4159-ad13-4f3b9804cceb-c000.csv"
df2 = pd.read_csv(data_path2, engine='python')

df = pd.concat([df1, df2], ignore_index=True)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["filtered_nouns_str"].values.astype('U'))

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

# 긍정단어 상위 10개
positive_top_10 = [feature_names[i] for i in coefficients.argsort()[-10:][::-1]]

# 부정단어 상위 10개
negative_top_10 = [feature_names[i] for i in coefficients.argsort()[:10]]

# 결과 출력
print("\nTop 10 Positive Words:")
positive_df = pd.DataFrame({'Word': positive_top_10, 'Coefficient': coefficients[coefficients.argsort()[-10:][::-1]]})
print(positive_df)

print("\nTop 10 Negative Words:")
negative_df = pd.DataFrame({'Word': negative_top_10, 'Coefficient': coefficients[coefficients.argsort()[:10]]})
print(negative_df)


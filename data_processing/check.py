import pandas as pd

# 로컬 파일에서 데이터프레임으로 읽어오기
local_file_path = '/home/maria_dev/BDP_TeamProject/data_processing/review.csv'
df = pd.read_csv(local_file_path)

# Score별 리뷰 개수 계산
score_counts = df['score'].value_counts().sort_index()

# 개수 출력
print("Score\tCount")
print("----------------")
for score, count in score_counts.items():
	    print(f"{score}\t{count}")


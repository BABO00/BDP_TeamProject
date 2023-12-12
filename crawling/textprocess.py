import pandas as pd

review = pd.read_csv("./kakaotalk_review.csv",encoding='utf-8-sig', index_col=0)
print(review)

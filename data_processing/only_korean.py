import sys
import pandas as pd
import numpy as np
import re

file_name = sys.argv[1]
df = pd.read_csv(file_name)
df = pd.DataFrame(df)
df['content'] = df['content'].fillna('')
df['content'] = df['content'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df.to_csv('review.csv',index=False)

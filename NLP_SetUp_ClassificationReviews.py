# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set seed
seed_value = 42
os.environ['Yelp_Review_NLP'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\Yelp_Reviews\Data'
os.chdir(path)

# Read data
df = pd.read_pickle('./YelpReviews_NLP.pkl')
print('Number of rows and columns:', df.shape)

# Subset columns for classification
df = df[['cleanReview', 'stars_reviews']]

# Recode to balance sets
df = df.copy()
df[['review_rank']] = df[['stars_reviews']]
df['review_rank'].mask(df['review_rank'] == 1, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 2, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 5, 1, inplace=True)

df1 = df[df.review_rank==1]
df1 = df1.sample(n=770743)

df2 = df[df.review_rank==0]

df = pd.concat([df1, df2])

del df1, df2

df = df.drop(['review_rank'], axis=1)

# Shuffle data
df = shuffle(df)

# Write parquet for colab 
df.to_parquet('YelpReviews_NLP_125stars_tokenized.parquet')

###############################################################################
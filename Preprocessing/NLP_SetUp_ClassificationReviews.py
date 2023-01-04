# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import warnings
import pandas as pd
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')
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

# Subset columns for classification
df = df[['cleanReview', 'stars_reviews']]

# Recode to balance sets
df[['review_rank']] = df[['stars_reviews']]
df['review_rank'].mask(df['review_rank'] == 1, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 2, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 5, 1, inplace=True)

# Filter and sample 5 star reviews
df1 = df[df.review_rank==1]
df1 = df1.sample(n=770743)

# Filter and sample 1 & 2 star reviews
df2 = df[df.review_rank==0]

# Concat and shuffle
df = pd.concat([df1, df2])
df = shuffle(df)
df = df.drop(['review_rank'], axis=1)

del df1, df2

# Write parquet for colab 
df.to_parquet('YelpReviews_NLP_125stars_tokenized.parquet')

###############################################################################
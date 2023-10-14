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
os.environ['YelpReviews_NLP'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\Yelp_Reviews\Data'
os.chdir(path)

# Read data
df = pd.read_pickle('./YelpReviews_NLP.pkl')

# Subset columns for classification
df = df[['cleanReview', 'stars_reviews', 'sentiment']]

# Recode to balance sets
df['stars'].mask(df['stars_reviews'] == 1, 0, inplace=True)
df['stars'].mask(df['stars_reviews'] == 2, 0, inplace=True)
df['stars'].mask(df['stars_reviews'] == 5, 1, inplace=True)

# Filter and sample 5 star reviews
df1 = df[df.stars==1]
df1 = df1.sample(n=414937)

# Filter and sample 1 & 2 star reviews
df2 = df[df.stars==0]
df2 = df2.sample(n=414937)

# Concat and shuffle
df = pd.concat([df1, df2])
df = shuffle(df)
df = df.drop(['stars'], axis=1)

del df1, df2

# Write parquet file
df.to_parquet('YelpReviews_NLP_125stars.parquet')

###############################################################################
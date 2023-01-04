# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import warnings
import pandas as pd
from textblob import TextBlob
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

# Find polarity using textblob
df['polarity'] = df['cleanReview'].apply(lambda x: TextBlob(x).sentiment[0])

# Examine descriptive stats of polarity
print(df['polarity'].describe())

# Define levels of sentiment to be used in classification
sentiment = []
for i in range(len(df)):
    if df['polarity'][i] >= 0.4:
        sentiment.append('Positive')
    if df['polarity'][i] > 0.2 and df['polarity'][i] < 0.4:
        sentiment.append('Slightly Positive')
    if df['polarity'][i] <= 0.2 and df['polarity'][i] >= 0.0:
        sentiment.append('Slightly Negative')
    if df['polarity'][i] < 0.0:
        sentiment.append('Negative')

# Join df
df = df.join(pd.DataFrame(sentiment))
df.rename(columns={0: 'sentiment'}, inplace=True)

# Examine how star reviews pairs with sentiment polarity
print(df[['stars_reviews', 'sentiment']].value_counts())

# Write processed reviews for sentiment + variables for later use
pd.to_pickle(df, './YelpReviews_NLP_sentiment.pkl')

# Recode sentiment for filtering out positve and negative polarity
df[['review_rank']] = df[['sentiment']]
df['review_rank'].mask(df['review_rank'] == 'Negative', 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 'Positive', 1, inplace=True)

# Sample equivalent size for balanced classes
df1 = df[df.review_rank==1]
df1 = df1.sample(n=414937)
df2 = df[df.review_rank==0]

# Concat and shuffle
df = pd.concat([df1, df2])
df = shuffle(df)
del df1, df2

df = df[['cleanReview', 'sentiment', 'stars_reviews']]
df.to_parquet('YelpReviews_NLP_sentimentNegPos_tokenized.parquet')

###############################################################################
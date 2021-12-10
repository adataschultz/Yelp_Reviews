# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import sys
import pandas as pd
import pickle

pd.set_option('display.max_columns', None)

# Set seed
seed_value = 42
os.environ['Yelp_Review_EDA'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('Yelp_Reviews_Preprocess.log.txt', 'w')

print('\nYelp Reviews Preprocess for Warehouse Construction') 
print('======================================================================')

# Set path
path = r'D:\Yelp_Reviews'
os.chdir(path)

# Define function for initial EDA
def data_summary(df):
    print('Number of Rows: {}, Columns: {}'.format(df.shape[0] ,df.shape[1]))
    a = pd.DataFrame()
    a['Number of Unique Values'] = df.nunique()
    a['Number of Missing Values'] = df.isnull().sum()
    a['Data type of variable'] = df.dtypes
    print(a)

print('\nYelp Reviews - Initial Summary') 
print('\n')
reviews_json_path = 'yelp_academic_dataset_review.json'
df_reviews = pd.read_json(reviews_json_path, lines=True)
df_reviews = df_reviews.drop_duplicates()

print(data_summary(df_reviews))
print('======================================================================')

print('\nYelp Business - Initial Summary') 
print('\n')
business_json_path = 'yelp_academic_dataset_business.json'
df_business = pd.read_json(business_json_path, lines=True)

# Convert dictionaries to strings for dropping duplicates after join
df_business['attributes'] = df_business['attributes'].astype(str)
df_business['hours'] = df_business['hours'].astype(str)
df_business =  df_business.drop_duplicates()

print(data_summary(df_business))
print('======================================================================')

print('\nYelp User - Initial Summary') 
print('\n')
user_json_path = 'yelp_academic_dataset_user.json'
df_user = pd.read_json(user_json_path, lines=True)
df_user = df_user.drop_duplicates()

print(data_summary(df_user))
print('======================================================================')

print('\nYelp Tip - Initial Summary') 
print('\n')
tip_json_path = 'yelp_academic_dataset_tip.json'
df_tip = pd.read_json(tip_json_path, lines=True)
df_tip = df_tip.drop_duplicates()

print(data_summary(df_tip))
print('======================================================================')

print('\nYelp Checkin - Initial Summary') 
print('\n')
checkin_json_path = 'yelp_academic_dataset_Checkin.json'
df_checkin = pd.read_json(checkin_json_path, lines=True)
df_checkin = df_checkin.drop_duplicates()

print(data_summary(df_checkin))
print('======================================================================')

##############################################################################
print('\nProcess Yelp Review and Business for data warehouse construction:')

# Convert keys to proper format for join of dfs
df_reviews['business_id'] = df_reviews['business_id'].astype(str)
df_business['business_id'] = df_business['business_id'].astype(str)

# Rename variables so unique for Reviews data
df_reviews.rename(columns = {'text': 'text_reviews'}, inplace = True) 
df_reviews.rename(columns = {'stars': 'stars_reviews'}, inplace = True) 
df_reviews.rename(columns = {'date': 'date_reviews'}, inplace = True) 
df_reviews.rename(columns = {'useful': 'useful_reviews'}, inplace = True) 
df_reviews.rename(columns = {'funny': 'funny_reviews'}, inplace = True) 
df_reviews.rename(columns = {'cool': 'cool_reviews'}, inplace = True) 

# Rename variables so unique for Business data
df_business.rename(columns = {'stars': 'stars_business'}, inplace = True) 
df_business.rename(columns = {'name': 'name_business'}, inplace = True) 
df_business.rename(columns = {'review_count': 'review_countbusiness'},
                   inplace = True) 
df_business.rename(columns = {'attributes': 'attributes_business'}, 
                   inplace = True) 
df_business.rename(columns = {'categories': 'categories_business'}, 
                   inplace = True) 
df_business.rename(columns = {'hours': 'hours_business'}, inplace = True) 

# Join reviews and business data for most complete set
df = pd.merge(df_reviews, df_business, how='right', left_on=['business_id'],
              right_on=['business_id'])
df = df.drop_duplicates()  

del df_reviews

print('\nSummary - Reviews joined with Business:')
print(data_summary(df))
print('======================================================================')

# Select rows that contain the most complete data
df = df[df.categories_business.notna()]

print('\nDimensions after finding most complete cases:', df.shape)
print('======================================================================')

##############################################################################
print('\nProcess Yelp User for data warehouse construction:')

# Drop user variables not using
df_user = df_user.drop(['name', 'friends', 'fans', 'compliment_photos'], axis=1)

# Rename columns so variables remain unique for User data
df_user.rename(columns = {'review_count': 'review_count_user'}, 
               inplace = True) 
df_user.rename(columns = {'yelping_since': 'yelping_since_user'}, 
               inplace = True) 
df_user.rename(columns = {'userful': 'useful_user'}, inplace = True) 
df_user.rename(columns = {'funny': 'funny_user'}, inplace = True) 
df_user.rename(columns = {'cool': 'cool_user'}, inplace = True) 
df_user.rename(columns = {'elite': 'eliter_user'}, inplace = True) 
df_user.rename(columns = {'average_stars': 'average_stars_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_hot': 'compliment_hot_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_more': 'compliment_more_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_profile': 'compliment_profile_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_cute': 'compliment_cute_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_list': 'compliment_list_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_note': 'compliment_note_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_plain': 'compliment_plain_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_cool': 'compliment_cool_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_funny': 'compliment_funny_user'},
               inplace = True) 
df_user.rename(columns = {'compliment_writer': 'compliment_writer_user'},
               inplace = True) 

# Join main table with User data
df = pd.merge(df, df_user, how='left', left_on=['user_id'],
              right_on=['user_id'])
df = df.drop_duplicates()  

del df_user

print('\nSummary - Reviews + Business joined with User:')
print(data_summary(df))
print('======================================================================')

##############################################################################
print('\nProcess Yelp Tip for data warehouse construction:')

# Drop vars not using
df_tip = df_tip.drop(['user_id', 'text'], axis=1)

# Rename variables in Tip data so unique and confirm type for join
df_tip.rename(columns = {'date': 'date_tip'}, inplace = True) 
df_tip.rename(columns = {'compliment_count': 'compliment_count_tip'}, 
              inplace = True) 

# Use Business data to get name of business in Tip table
df_tip['name_business'] = df_tip['business_id'].map(df_business.set_index('business_id')['name_business'])

del df_business

# Get sum of compliment count for each business id
df_tip1 = df_tip.groupby('business_id')['compliment_count_tip'].sum().reset_index()
df_tip1.rename(columns = {'compliment_count_tip': 'compliment_count_tip_idSum'},
               inplace = True) 

# Merge back for sum of compliments for each id
df_tip = pd.merge(df_tip, df_tip1, how='left', left_on=['business_id'],
              right_on=['business_id'])
df_tip = df_tip.drop_duplicates()  

del df_tip1

# Get sum of compliment count for each business name
df_tip1 = df_tip.groupby('name_business')['compliment_count_tip'].sum().reset_index()
df_tip1.rename(columns = {'compliment_count_tip': 'compliment_count_tip_businessSum'},
               inplace = True) 

# Merge back for sum of compliments for each business name
df_tip = pd.merge(df_tip, df_tip1, how='left', left_on=['name_business'],
              right_on=['name_business'])
df_tip = df_tip.drop_duplicates()  

del df_tip1

# Drop features used for feature engineering and duplicates
df_tip = df_tip.drop(['date_tip', 'compliment_count_tip'], axis=1)
df_tip = df_tip.drop_duplicates(subset = ['business_id'])

print('\nSummary - Tip after Compliment Count Sums:')
print(data_summary(df_tip))
print('======================================================================')

# Convert keys to proper format for join of dfs
df_tip['business_id'] = df_tip['business_id'].astype(str)
df_tip['name_business'] = df_tip['name_business'].astype(str)

df = df.copy()
df['name_business'] = df['name_business'].astype(str)

# Join main table with Tip data containing compliment counts
df = pd.merge(df, df_tip, how='right', left_on=['business_id', 'name_business'],
              right_on=['business_id', 'name_business'])
df = df.drop_duplicates()  

del df_tip

print('\nSummary - Reviews + Business + User joined with Tip:')
print(data_summary(df))
print('======================================================================')
# Select rows that contain the most complete data
df = df[df.review_id.notna()]

print('\nSummary - Complete cases of Reviews + Business + User + Tip:')
print(data_summary(df))
print('======================================================================')

##############################################################################
print('\nProcess Yelp Checkins for data warehouse construction:')

# Process time variables
df_checkin['business_id'] = df_checkin['business_id'].astype(str)
df_checkin.rename(columns = {'date': 'businessCheckin_date'},
               inplace = True) 

# Extract each date as a row for each business id
df_checkin1 = df_checkin.set_index(['business_id']).apply(lambda x: x.str.split(',').explode()).reset_index()                                                

# Convert date to datetime for extracting time variables
df_checkin1 = df_checkin1.copy()
df_checkin1['businessCheckin_date'] = pd.to_datetime(df_checkin1['businessCheckin_date'], 
                                     format='%Y-%m-%d %H:%M:%S', errors='ignore')

df_checkin1['businessCheckin_Year'] = df_checkin1.businessCheckin_date.dt.year
df_checkin1['businessCheckin_YearMonth'] = df_checkin1['businessCheckin_date'].dt.to_period('M')
df_checkin1['businessCheckin_YearWeek'] = df_checkin1['businessCheckin_date'].dt.strftime('%Y-w%U')

# Create hourly variables
df_checkin1['businessCheckin_hourNumber'] = df_checkin1.businessCheckin_date.dt.hour

# Create PM/AM time in the day
mask = df_checkin1['businessCheckin_hourNumber'] >= 12
df_checkin1.loc[mask, 'businessCheckin_hourNumber'] = 'PM'

mask = df_checkin1['businessCheckin_hourNumber'] !='PM'
df_checkin1.loc[mask, 'businessCheckin_hourNumber'] = 'AM'

# Get count of date for each checkin for each business id
df_checkin2 = df_checkin1.groupby('business_id')['businessCheckin_date'].count().reset_index()
df_checkin2.rename(columns = {'businessCheckin_date': 'dateDay_checkinSum'},
               inplace = True) 
df_checkin2 = df_checkin2.drop_duplicates()

# Merge back for count of checkin dates for business id
df_checkin = pd.merge(df_checkin, df_checkin2, how='left', left_on=['business_id'],
              right_on=['business_id'])
df_checkin = df_checkin.drop_duplicates()  

del df_checkin2

# Get count of Year for each checkin for each business id
df_checkin2 = df_checkin1.groupby('business_id')['businessCheckin_Year'].count().reset_index()
df_checkin2.rename(columns = {'businessCheckin_Year': 'dateYear_checkinSum'},
               inplace = True) 
df_checkin2 = df_checkin2.drop_duplicates()

# Merge back for count of checkin Years for business id
df_checkin = pd.merge(df_checkin, df_checkin2, how='left', left_on=['business_id'],
              right_on=['business_id'])
df_checkin = df_checkin.drop_duplicates()  

del df_checkin2

# Get count of YearMonth for each checkin for each business id
df_checkin2 = df_checkin1.groupby('business_id')['businessCheckin_YearMonth'].count().reset_index()
df_checkin2.rename(columns = {'businessCheckin_YearMonth': 'dateYearMonth_checkinSum'},
               inplace = True) 
df_checkin2 = df_checkin2.drop_duplicates()

# Merge back for count of checkin YearMonth for business id
df_checkin = pd.merge(df_checkin, df_checkin2, how='left', left_on=['business_id'],
              right_on=['business_id'])
df_checkin = df_checkin.drop_duplicates()  

del df_checkin2

# Get count of YearWeek for each checkin for each business id
df_checkin2 = df_checkin1.groupby('business_id')['businessCheckin_YearWeek'].count().reset_index()
df_checkin2.rename(columns = {'businessCheckin_YearWeek': 'dateYeaWeek_checkinSum'},
               inplace = True) 
df_checkin2 = df_checkin2.drop_duplicates()

del df_checkin1

# Merge back for count of checkin YearWeek for business id
df_checkin = pd.merge(df_checkin, df_checkin2, how='left', left_on=['business_id'],
              right_on=['business_id'])
df_checkin = df_checkin.drop_duplicates()  

del df_checkin2

# Join main table with Checkin data
df = pd.merge(df, df_checkin, how='left', left_on=['business_id'],
              right_on=['business_id'])
df = df.drop_duplicates()  

del df_checkin

print('\nSummary - Reviews + Business + User + Tip joined with Checkin:')
print(data_summary(df))
print('======================================================================')

# Select rows that contain the most complete data
df = df[df.businessCheckin_date.notna()]

print('\nSummary - Complete Cases of Reviews + Business + User + Tip + Checkin:')
print(data_summary(df))
print('======================================================================')

# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
# Write warehouse to pickle file
pd.to_pickle(df, './211209_YelpReviews.pkl')

##############################################################################
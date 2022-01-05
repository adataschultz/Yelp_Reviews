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
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set resolution of saved graphs
my_dpi = 96

# Set seed
seed_value = 42
os.environ['Yelp_Review_EDA'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\Yelp_Reviews\Data'
os.chdir(path)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('Yelp_Reviews_Preprocess.log.txt', 'w')

print('\nYelp Reviews Preprocess for Warehouse Construction') 
print('======================================================================')

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
df_user.rename(columns = {'useful': 'useful_user'}, inplace = True) 
df_user.rename(columns = {'funny': 'funny_user'}, inplace = True) 
df_user.rename(columns = {'cool': 'cool_user'}, inplace = True) 
df_user.rename(columns = {'elite': 'elite_user'}, inplace = True) 
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

mask = df_checkin1['businessCheckin_hourNumber'] != 'PM'
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
df_checkin2.rename(columns = {'businessCheckin_YearWeek': 'dateYearWeek_checkinSum'},
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

# Process categories 
df_category_split = df[['business_id', 'categories_business']]
df_category_split.set_index('business_id')

# Create new column for each string after a comma
df_category_split = df_category_split.copy()
df_category_split['categories_business'] = df_category_split['categories_business'].str.split(',', 
                                                                                              expand=True)
           
# Rename column for unique name
df_category_split.rename(columns = {'categories_business': 'categories_combined'},
               inplace = True) 

# Remove first space to align each observation
df_category_split = df_category_split.copy()
df_category_split['categories_combined'] = df_category_split['categories_combined'].str.strip()
df_category_split = df_category_split.drop_duplicates()

# Join main table with processed Category
df = pd.merge(df, df_category_split, how='right', left_on=['business_id'],
              right_on=['business_id'])
df = df.drop(['categories_business'], axis=1)
df = df.drop_duplicates()  

del df_category_split

print('\nSummary - Preprocessing Yelp Reviews for Category:')
print(data_summary(df))
print('======================================================================')

# Process business hours to non-dictionary by subsetting id with hours
df1 = df[['business_id', 'hours_business']]

df1 = df1.copy()
df1.loc[:,'hours_businessOriginal'] = df1['hours_business']

# Extract each day and hours open/closed as a row for each business id
df1 = df1.set_index(['business_id', 'hours_businessOriginal']).apply(lambda x: x.str.split(',').explode()).reset_index()   
  
df1 = df1.drop_duplicates()
df1 = df1[df1['business_id'].notna()]

print('\nDimensions of Exploded Business Open/Closed Days/Hours:', df1.shape) 
print('======================================================================')

# Process dictionary further for regex characteristics
# Remove brackets from the initial dictionary
df1 = df1.copy()
df1['hours_business'] = df1['hours_business'].astype(str)
df1.loc[:,'hours_business'] =  df1['hours_business'].str.replace(r'{', '', 
                                                                 regex=True)
df1.loc[:,'hours_business'] =  df1['hours_business'].str.replace(r'}', '', 
                                                                 regex=True)

# Set original business dictionary in index so format is retained
df1 = df1.set_index(['hours_businessOriginal'])

# Remove single quotes
df1 = df1.replace("'", "", regex=True)

# Remove first space to align each observation
df1['hours_business'] = df1['hours_business'].str.strip()
df1 = df1.drop_duplicates()

# Find how many unique business day/hours
print('\nNumber of Unique business hours after removing Regex characteristics:',
      df1[['hours_business']].nunique()) 
print('======================================================================') 

# Reset index of df
df1 = df1.reset_index()

# Filter businesses with no hours for open or closed
df2 = df1[(df1['hours_business'] == 'None')]

# Process needed variables for businesses that have open/closed for concatenation
df2 = df2.copy()
df2.loc[:,'hoursBusiness_day'] = 'NA'
df2.loc[:,'hours_working'] = 'NA'
df2.loc[:,'hours_businessOpen'] = 'NA'
df2.loc[:,'hours_businessOpen2'] = 'NA'
print('\nDimensions of businesses with no open/closed hours:', df2.shape) 
print('======================================================================') 

# Filter businesses with hours provided
df3 = df1[(df1['hours_business'] != 'None')]
print('\nDimensions of businesses with no open/closed hours:', df3.shape)  
print('======================================================================') 

# Extract day from the business hours
df3 = df3.copy()
df3['hoursBusiness_day'] = df3['hours_business'].str.rsplit(':').str[0] 

# Create new variable of everything after 'y' in day for open/closed time
df3['hours_working'] = df3.hours_business.str.extract('y:(.*)')  

# Extract when the business opens
df3['hours_businessOpen'] = df3['hours_working'].str.rsplit('-').str[0] 

# Extract the time before the : when the business opens
df3['hours_businessOpen1:'] = df3['hours_business'].str.rsplit(':').str[-3] 

# Extract the time after whole hours in minutes
df3['hours_businessOpen1'] = df3['hours_businessOpen'].str.rsplit(':').str[-1] 

# Subset business opening times that are whole hours
df3['hours_businessOpen1:'] = df3['hours_businessOpen1:'].astype(int)

df4 = df3.loc[(df3['hours_businessOpen1:'] == 0)]
df4 = df4.drop(['hours_businessOpen1', 'hours_businessOpen1:'], axis=1)
df4 = df4.copy()

# Define midnight as '0' for military time
df4.loc[:,'hours_businessOpen2'] = 0
print('\nDimensions of businesses with open hours at Midnight:', df4.shape) 
print('======================================================================') 

# Process time for non whole hour minutes
df5 = df3.loc[(df3['hours_businessOpen1:'] != 0)]

del df3

# Convert non whole hours to integer for creation of true business hours
df5 = df5.copy()
df5['hours_businessOpen1'] = df5['hours_businessOpen1'].astype(int)
df5['hours_businessOpen1'] = df5['hours_businessOpen1'] / 60

# Create an updated time for businesses that are open on non whole hours
df5['hours_businessOpen1:'] = df5['hours_businessOpen1:'].astype(int)
df5['hours_businessOpen2'] = df5['hours_businessOpen1:'] + df5['hours_businessOpen1']

df5 = df5.drop(['hours_businessOpen1', 'hours_businessOpen1:'], axis=1)
print('\nDimensions of businesses with non whole open hours:', df5.shape) 
print('======================================================================') 

# Concatenate sets with no business open hours whole and non whole hours 
data = [df2, df4, df5]
df7 = pd.concat(data)
print('\nDimensions of businesses with no & business hours open modified:',
      df7.shape)
print('======================================================================') 

del data, df2, df4, df5

###############################################################################
# Filter businesses with no hours for creating when business closes
df1 = df7[(df7['hours_business'] == 'None')]

df1 = df1.copy()
df1.loc[:,'hours_businessClosed'] = 'NA'
df1.loc[:,'hours_businessClosed2'] = 'NA'

# Filter businesses with hours provided
df2 = df7[(df7['hours_business'] != 'None')]

del df7

# Extract when the business closes after the '-'
df2 = df2.copy()
df2['hours_businessClosed'] = df2['hours_business'].str.rsplit('-').str[1] 

# Extract the time before the : when the business closes
df2['hours_businessClosed1:'] = df2['hours_business'].str.rsplit('-').str[1] 
df2['hours_businessClosed1:'] = df2['hours_businessClosed1:'].str.rsplit(':').str[0] 

# Extract when the business closes after the whole hour for minutes
df2['hours_businessClosed1'] = df2['hours_businessClosed'].str.rsplit(':').str[-1] 

# Subset business closing times that are midnight and whole hours
df2 = df2.copy()
df2['hours_businessClosed1:'] = df2['hours_businessClosed1:'].astype(int)
df2['hours_businessClosed1'] = df2['hours_businessClosed1'].astype(int)

df3 = df2.loc[(df2['hours_businessClosed1:'] == 0) & (df2['hours_businessClosed1'] == 0)]

df3 = df3.drop(['hours_businessClosed1', 'hours_businessClosed1:'], axis=1)
df3.loc[:,'hours_businessClosed2'] = 0
print('\nDimensions of businesses with closing times that are midnight and whole hours:',
      df3.shape)
print('======================================================================') 

# Subset business closing times that are midnight and non whole hours
df4 = df2.loc[(df2['hours_businessClosed1:'] == 0) & (df2['hours_businessClosed1'] != 0)]

df4 = df4.copy()
df4['hours_businessClosed1:'] = df4['hours_businessClosed1:'].astype(int)
df4['hours_businessClosed1'] = df4['hours_businessClosed1'].astype(int)
df4['hours_businessClosed1'] = df4['hours_businessClosed1'] / 60

df4['hours_businessClosed2'] = df4['hours_businessClosed1:'] + df4['hours_businessClosed1']

df4 = df4.drop(['hours_businessClosed1', 'hours_businessClosed1:'], axis=1)
print('\nDimensions of businesses with closing times that are midnight and non whole hours:',
      df4.shape) 
print('======================================================================') 

# All non midnight closing on whole hours
df5 = df2.loc[(df2['hours_businessClosed1:'] != 0) & (df2['hours_businessClosed1'] == 0)]

df5 = df5.copy()
df5['hours_businessClosed1:'] = df5['hours_businessClosed1:'].astype(int)
df5['hours_businessClosed1'] = df5['hours_businessClosed1'].astype(int)

df5['hours_businessClosed2'] = df5['hours_businessClosed1:'] + df5['hours_businessClosed1']

df5 = df5.drop(['hours_businessClosed1', 'hours_businessClosed1:'], axis=1)
print('\nDimensions of businesses with closing times that are non midnight and whole hours:',
      df5.shape) 
print('======================================================================') 

# All non midnight closing not on whole hours
df6 = df2.loc[(df2['hours_businessClosed1:'] != 0) & (df2['hours_businessClosed1'] != 0)]

df6 = df6.copy()
df6['hours_businessClosed1:'] = df6['hours_businessClosed1:'].astype(int)
df6['hours_businessClosed1'] = df6['hours_businessClosed1'].astype(int)
df6['hours_businessClosed1'] = df6['hours_businessClosed1'] / 60

df6['hours_businessClosed2'] = df6['hours_businessClosed1:'] + df6['hours_businessClosed1']

df6 = df6.drop(['hours_businessClosed1', 'hours_businessClosed1:'], axis=1)
print('\nDimensions of businesses with closing times that are non midnight and non whole hours:',
      df6.shape) 
print('======================================================================') 

# Concatenate sets with no business closing hours whole and non whole hours 
data = [df1, df3, df4, df5, df6]
df1 = pd.concat(data)

df1 = df1.drop(['hours_businessOpen', 'hours_businessClosed'], axis=1)
df1 = df1.drop_duplicates()

print('\nDimensions of businesses with no & business hours closed modified:',
      df1.shape)
print('======================================================================') 

del data, df2, df3, df4, df5, df6

# Rename created variables
df1.rename(columns={'hours_businessOpen2':'hours_businessOpen',
                    'hours_businessClosed2':'hours_businessClosed'}, 
           inplace=True)

# Find businesses who do not have hours for open or closed
df2 = df1[(df1['hours_working'] == 'NA')]

# Feature engineer variables for concatenation
df2 = df2.copy()
df2.loc[:,'hours_businessOpenDaily'] = 'NA'
df2.loc[:,'hours_businessOpen_amPM'] = 'NA'
df2.loc[:,'hours_businessClosed_amPM'] = 'NA'

# Find businesses who have business hours that are not missing
df1 = df1[(df1['hours_working'] != 'NA')]

# Create feature for number of hours each day
df1 = df1.copy()
df1['hours_businessOpen'] = df1['hours_businessOpen'].astype('float64')
df1['hours_businessClosed'] = df1['hours_businessClosed'].astype('float64')
df1['hours_businessOpenDaily'] = df1['hours_businessClosed'] - df1['hours_businessOpen']

# Create PM/AM time in the day for when business opens
mask = df1['hours_businessOpen'] >= 12
df1.loc[mask, 'hours_businessOpen_amPM'] = 'PM'

mask = df1['hours_businessOpen_amPM'] !='PM'
df1.loc[mask, 'hours_businessOpen_amPM'] = 'AM'

# Create PM/AM time in the day for when business closes
mask = df1['hours_businessClosed'] >= 12
df1.loc[mask, 'hours_businessClosed_amPM'] = 'PM'

mask = df1['hours_businessClosed_amPM'] !='PM'
df1.loc[mask, 'hours_businessClosed_amPM'] = 'AM'

# Concatenate sets with NA business hours and ones with data
df1 = pd.concat([df1, df2])

# Rename processed business hours since Regex allowed for new vars
df1.rename(columns={'hours_business': 'hours_businessRegex'}, inplace=True)

# Join main table with processed business hours variables
df = df.copy()
df['business_id'] = df['business_id'].astype(str)
df['hours_business'] = df['hours_business'].astype(str)

df1 = df1.copy()
df1['business_id'] = df1['business_id'].astype(str)
df1['hours_businessOriginal'] = df1['hours_businessOriginal'].astype(str)

# Join main table with business hours table
df = pd.merge(df, df1, how='outer', left_on=['business_id', 'hours_business'], 
               right_on=['business_id', 'hours_businessOriginal'])
df = df.drop_duplicates(subset=['review_id'])

del df1

print('\nSummary - Preprocessing Yelp Reviews for Category:')
print(data_summary(df))
print('======================================================================')

###############################################################################
# TBD: Process attributes of the different businesses
###############################################################################
# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
########################### Exploratory Data Analysis #########################
###############################################################################
# Set path for EDA
path = r'D:\Yelp_Reviews\EDA'
os.chdir(path)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('Yelp_Reviews_EDA.log.txt', 'w')

print('\nYelp Reviews EDA') 
print('======================================================================')

# Subset quantitative vars from reviews and business for graphing
df_num = df[['stars_reviews', 'useful_reviews', 'funny_reviews', 'cool_reviews',
             'stars_business', 'review_countbusiness', 'is_open']]

print('\nDescriptive statistics of quant vars in Reviews + Business:')
print(df_num.describe(include = [np.number]).round(2))
print('======================================================================')

# Drop is_open since binary
df_num1 = df_num.drop(['is_open'], axis=1)

# Hist plots of quant vars
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Histograms of Quantitative Variables in Yelp Reviews & Businesses',
             fontsize=20)
for variable, subplot in zip(df_num1, ax.flatten()):
    sns.histplot(x=df_num1[variable], kde=True, ax=subplot)
plt.tight_layout()  
fig.savefig('EDA_Quant_Histplot_ReviewsBusiness_output.png', dpi=my_dpi * 10,
            bbox_inches='tight') 

# Box-and-whisker plots of quant vars
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Boxplots of Quantitative Variables in Yelp Reviews & Businesses', 
             fontsize=20)
for var, subplot in zip(df_num1, ax.flatten()):
  sns.boxplot(x=df_num1[var], data=df_num1, ax=subplot)
plt.tight_layout()
fig.savefig('EDA_Quant_Boxplots_ReviewsBusiness_output.png', dpi=my_dpi * 10,
            bbox_inches='tight')

del df_num1

# Count plot to examine quant vars
df_num2 = df_num.drop(['useful_reviews', 'funny_reviews', 'cool_reviews'],
                      axis=1)

fig, ax = plt.subplots(1, 4, figsize=(15, 10))
fig.suptitle('Countplots of Quantitative Variables in Yelp Reviews & Businesses') 
for var, subplot in zip(df_num2, ax.flatten()): 
    sns.countplot(x=df_num2[var], ax=subplot)
plt.tight_layout()  
fig.savefig('EDA_Quant_Countplot_ReviewsBusiness_output.png', dpi=my_dpi * 10,
            bbox_inches='tight')

del df_num2

###############################################################################
# Examine reviews based on variables in original Reviews data
total_reviews = len(df)
useful_reviews = len(df[df['useful_reviews'] > 0])
funny_reviews = len(df[df['funny_reviews'] > 0])
cool_reviews = len(df[df['cool_reviews' ] > 0])

# Determine cutoff for different quality reviews by summary stats
positive_reviews  = len(df[df['stars_reviews'] == 5.0])
negative_reviews = len(df[df['stars_reviews'] <= 2.5])
ok_reviews = len(df[df['stars_reviews'].between(3, 4.5)])

# Calculate percentage of reviews in each group
print('\nTotal reviews: {}'.format(total_reviews))
print('\nPercentage of useful reviews: ' +  str(round((useful_reviews/total_reviews)*100)))
print('\nPercentage of funny reviews: ' +  str(round((funny_reviews/total_reviews)*100)))
print('\nPercentage of cool reviews: ' +  str(round((cool_reviews/total_reviews)*100)))
print('\nPercentage of positive reviews: ' +  str(round((positive_reviews/total_reviews)*100)))
print('\nPercentage of negative reviews: ' +  str(round((negative_reviews/total_reviews)*100)))
print('\nPercentage of OK reviews: ' +  str(round((ok_reviews/total_reviews)*100)))
print('======================================================================')

# Find the top 30 restaurants by count for graphing
top30_restaurants = df.name_business.value_counts().index[:30].tolist()

top30_restaurantsCount = len(top30_restaurants)
total_businesses = df['name_business'].nunique()

print('\nPercentage of top 30 restaurants of total: ' +  str((top30_restaurantsCount/total_businesses)*100))

# Filter top 30 restaurants by count
top30_restaurants = df.loc[df['name_business'].isin(top30_restaurants)]

# Plot the mean stars on reviews in the top 30 restaurants 
print('\nAverage Review Rating of 30 Most Frequent Restaurants')
print(top30_restaurants.groupby(top30_restaurants.name_business)['stars_reviews'].mean().sort_values(ascending=False))
top30_restaurants.groupby(top30_restaurants.name_business)['stars_reviews'].mean().sort_values(ascending=True).plot(kind='barh', 
                                                                                                                    figsize=(12, 10))
plt.title('Average Review Rating of 30 Most Frequent Restaurants',
          fontsize=20)
plt.ylabel('Name of Restaurant', fontsize=18)
plt.xlabel('Average Review Rating', fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('AverageReviewRating_Top30RatedRestaurants_Yelp.png', 
            bbox_inches='tight', dpi=my_dpi * 10)
print('======================================================================')

# Plot the mean 'useful','funny', 'cool' reviews in the top 30 restaurants sorted by 'useful'
top30_restaurants.groupby(top30_restaurants.name_business)[['useful_reviews', 
                                                            'funny_reviews', 
                                                            'cool_reviews']].mean().sort_values('useful_reviews', 
                                                                                                ascending=True).plot(kind='barh',
                                                                                                                     figsize=(15, 14), 
                                                                                                                     width=0.7)
plt.title('Average Useful, Funny & Cool Reviews in the 30 Most Frequent Restaurants', 
          fontsize=28)
plt.ylabel('Name of Restaurant', fontsize=18)
plt.yticks(fontsize=20)
plt.legend(fontsize=22)
plt.savefig('UsefulFunnyCool_SortUseful_Top30RatedRestaurants_Yelp.png',
            bbox_inches='tight', dpi=my_dpi * 10)

###############################################################################
# Examine is_open to see which restaurant are still open
print('\nPercentage of business are open or closed: ')                                                       
print(df['is_open'].value_counts(normalize=True) * 100)
print('======================================================================')

###############################################################################
# Find the top 10 states with the highest number of reviews
x = df['state'].value_counts()[:10].to_frame()
print('\nTop 10 states with the highest number of reviews:')                                                       
print(x)
print('======================================================================')

sns.barplot(x=x['state'], y=x.index)
plt.title('States with the 10 highest number of business reviews listed in Yelp', 
          fontsize=16)
plt.xlabel('Counts of Reviews', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.tight_layout()  
plt.savefig('Top10States_MostReviews_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)

del x

# Find the top 30 cities with the highest number of reviews
# Set graph settings 
sns.set_style('whitegrid')

x = df['city'].value_counts()
x = x.sort_values(ascending=False)
x = x.iloc[0:30]
print('\nTop 30 cities with the highest number of reviews:')                                                       
print(x)

plt.figure(figsize=(16,6))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.9)
plt.title('Cities with the Highest Number of Reviews on Yelp', fontsize=18)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Name of City', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)

# Add text labels to graph
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 3, label, ha='center',
            va='bottom', fontsize=9)
plt.savefig('Cities_MostReviews_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)
del x

print('======================================================================')

# Find the top 30 cities with the highest number of reviews
top30_city = df.city.value_counts().index[:30].tolist()
top30_city = df.loc[df['city'].isin(top30_city)]

# Find the top 30 cities with the highest number of reviews and highest rating
print('\nTop 30 cities with the highest number of reviews and highest rating')
print(top30_city.groupby(top30_city.city)['stars_reviews'].mean().sort_values(ascending=True))

top30_city.groupby(top30_city.city)['stars_reviews'].mean().sort_values(ascending=True).plot(kind='barh',
                                                                                             figsize=(12, 10))
plt.yticks(fontsize=18)
plt.title('Average Review Rating of Cities with Highest Number of Reviews on Yelp', 
          fontsize=20)
plt.ylabel('Name of City', fontsize=18)
plt.xlabel('Average Review Rating', fontsize=18)
plt.tight_layout()  
plt.savefig('AverageReviewRating_Top30Cities_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)
print('======================================================================')

###############################################################################
# Convert review date to datetime for extracting time variables
df = df.copy()
df['date_reviews'] = pd.to_datetime(df['date_reviews'], 
                                     format='%Y-%m-%d %H:%M:%S', errors='ignore')
df['date_reviews_Year'] = df.date_reviews.dt.year
df['date_reviews_YearMonth'] = df['date_reviews'].dt.to_period('M')
df['date_reviews_YearWeek'] = df['date_reviews'].dt.strftime('%Y-w%U')

# Most recent Trending businesses
top = 5
temp = df[['business_id', 'date_reviews_Year', 'stars_business']]
five_star_reviews = temp[temp['stars_business']==5]
trendy = five_star_reviews.groupby(['business_id',
                                    'date_reviews_Year']).size().reset_index(name='counts')

trending = trendy.sort_values(['date_reviews_Year',
                               'counts'])[::-1][:top].business_id.values
for business_id in trending:
    record = trendy.loc[trendy['business_id'] == business_id]
    business_name = df.loc[df['business_id'] == business_id].name_business.values[0]
    series = pd.Series(record['counts'].values,
                       index=record.date_reviews_Year.values,
                       name='Trending business')
    axes = series.plot(kind="bar",figsize=(10, 7))
    plt.xlabel('Year', axes=axes)
    plt.ylabel('Total positive reviews', axes=axes)
    plt.title('Review trend of {}'.format(business_name), axes=axes)    
    plt.savefig('Top5Trending'+'{}.png'.format(business_id),
                bbox_inches='tight')  
    
###############################################################################
# Examine categories
x = df.categories_combined.value_counts()
print('There are', len(x), 'different categories of Businesses in Yelp')
print('\n')

x = x.sort_values(ascending=False)
x = x.iloc[0:20]
print('\nTop 20 categories in Yelp:')
print(x)
print('======================================================================')

plt.figure(figsize=(16,6))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.9)
plt.title('Top 20 Categories in Yelp Reviews', fontsize=18)
locs, labels = plt.xticks()
plt.setp(labels, rotation=70)
plt.ylabel('Number of Businesses', fontsize=14)
plt.xlabel('Type of Category', fontsize=14)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center',
            va='bottom', fontsize=11)
plt.tight_layout()  
plt.savefig('Top20Categories_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)

del x

###############################################################################
# Examine user information
df_num = df[['review_count_user', 'useful_user', 'funny_user', 'cool_user',
                  'average_stars_user', 'compliment_hot_user', 
                  'compliment_more_user','compliment_profile_user', 
                  'compliment_cute_user', 'compliment_list_user', 
                  'compliment_note_user', 'compliment_plain_user', 
                  'compliment_cool_user', 'compliment_funny_user', 
                  'compliment_writer_user']]

print('\nDescriptive statistics of quant vars in Yelp User:')
print(df_num.describe(include = [np.number]).round(2))
print('======================================================================')

# Box-and-whisker plots of quant vars in User
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('Boxplots of Quantitative Variables in Users on Yelp Reviews')
for var, subplot in zip(df_num, ax.flatten()):
  sns.boxplot(x=df_num[var], data=df_num, ax=subplot)
plt.tight_layout()
fig.savefig('EDA_Quant_Boxplots_User_output.png', dpi=my_dpi * 10,
            bbox_inches='tight')

# Examine the number of reviews by each user
sns.histplot(x='review_count_user', data=df_num, kde=True)
plt.ylim(0, 100000)
plt.title('Count of reviews by Users in Yelp')
plt.tight_layout()  
plt.savefig('CountReviews_Users_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)

# Examine the average rating by each user
sns.histplot(x='average_stars_user', data=df_num, kde=True)
plt.ylim(0, 300000)
plt.title('Count of Average Stars by Users in Yelp')
plt.tight_layout()  
plt.savefig('AverageStars_Users_Yelp.png', bbox_inches='tight',
            dpi=my_dpi * 10)

# Filter characteristics of users for graphing
df_num1 = df_num[['useful_user', 'funny_user', 'cool_user']]

# Histplots of quant vars in User
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Histograms of Quantitative Variables in Users on Yelp Reviews') 
for var, subplot in zip(df_num1, ax.flatten()):
    sns.histplot(x=df_num1[var], kde=True, ax=subplot)
    plt.ylim(0, 50000)
plt.tight_layout()  
fig.savefig('EDA_Quant_Histplot_User_3var_output.png', dpi=my_dpi * 10,
            bbox_inches='tight') 

del df_num1

# Filter compliment types of users for graphing
df_num2 = df_num[['compliment_more_user','compliment_profile_user', 
                  'compliment_cute_user', 'compliment_list_user', 
                  'compliment_note_user', 'compliment_plain_user', 
                  'compliment_cool_user', 'compliment_funny_user', 
                  'compliment_writer_user']]

# Histplots of quant vars in User
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Histograms of Quantitative Variables in Users on Yelp Reviews') 
for var, subplot in zip(df_num2, ax.flatten()):
    sns.histplot(x=df_num2[var], kde=True, ax=subplot)
plt.tight_layout()  
fig.savefig('EDA_Quant_Histplot_UserCompliment_output.png',
            bbox_inches='tight')

del df_num2

# Count plot to examine quant vars
df_num3 = df[['compliment_count_tip_idSum', 'compliment_count_tip_businessSum']]

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Countplots of Compliment Counts for Business ID and Total Business') 
for var, subplot in zip(df_num3, ax.flatten()): 
    sns.countplot(x=df_num3[var], ax=subplot)
plt.tight_layout()  
fig.savefig('EDA_Quant_Countplot_ComplimentCountSumBusiness_output.png', 
            dpi=my_dpi * 10, bbox_inches='tight')

del df_num3, df_num

# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
# Write warehouse to pickle file
pd.to_pickle(df, './220104_YelpReviews.pkl')

##############################################################################
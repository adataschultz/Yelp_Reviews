# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import pandas as pd
import pickle
import time 
from datetime import datetime, timedelta
import dask.dataframe as dd
import re
import string, unicodedata
import contractions
import spacy
import langid
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import matplotlib.pyplot as plt
from nltk import FreqDist
import csv
from wordcloud import WordCloud,STOPWORDS

# Download word sets for cleaning
nlp = spacy.load('en_core_web_lg')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

my_dpi = 96

# Set seed
seed_value = 42
os.environ['Yelp_Review_NLP'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\Yelp_Reviews\Data'
os.chdir(path)

print('\nYelp Reviews NLP Preprocessing') 
print('======================================================================')

# Read data
df = pd.read_pickle('./YelpReviews.pkl')
print('Number of rows and columns:', df.shape)

# Subset variables of interest for reviews
df = df[['review_id', 'text_reviews', 'stars_reviews', 'name_business', 'city',
         'state', 'stars_business', 'review_countbusiness', 
         'categories_combined']]

###############################################################################
# Initial EDA of reviews
# Set path
path = r'D:\Yelp_Reviews\EDA\NLP'
os.chdir(path)

# Word count for each review
df['review_wordCount'] = df['text_reviews'].apply(lambda x: len(str(x).split()))

# Character count for each review
df['review_charCount'] = df['text_reviews'].apply(lambda x: len(str(x)))

# Length of words and characters in each initial review
f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(15, 10), sharex=True)
f.suptitle('Yelp Reviews: Length of Words and Characters for Each Review',
           fontsize = 25)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=30)
#f.supxlabel('Review Number')
ax1.plot(df['review_wordCount'], color = 'red')
ax1.set_ylabel('Word Count', fontsize=20)
ax2.plot(df['review_charCount'], color = 'blue')
ax2.set_ylabel('Character Count', fontsize=20)

f.savefig('lengthInitialReviews.png', dpi=my_dpi, 
          bbox_inches='tight')

df = df.drop(['review_wordCount', 'review_charCount'], axis=1)

###############################################################################
# Filter data for text analysis of restaurants
# Filter states with the 7 highest counts of reviews
df1 = df['state'].value_counts().index[:7]
df = df[df['state'].isin(df1)]

del df1

print('\nDimensions after filtering US states with the 7 highest count of reviews:',
      df.shape) #(7704864, 10)
print('======================================================================') 

# Filter food categories with over 30k counts
df = df.loc[(df['categories_combined']=='Restaurants') 
             | (df['categories_combined']=='Food')
             | (df['categories_combined']=='American (New)') 
             | (df['categories_combined']=='American (Traditional)')
             | (df['categories_combined']=='Pizza') 
             | (df['categories_combined']=='Sandwiches')
             | (df['categories_combined']=='Breakfast & Brunch') 
             | (df['categories_combined']=='Mexican')
             | (df['categories_combined']=='Italian') 
             | (df['categories_combined']=='Seafood')
             | (df['categories_combined']=='Japanese')
             | (df['categories_combined']=='Burgers')
             | (df['categories_combined']=='Sushi Bars')
             | (df['categories_combined']=='Chinese')
             | (df['categories_combined']=='Desserts')
             | (df['categories_combined']=='Thai')
             | (df['categories_combined']=='Bakeries')
             | (df['categories_combined']=='Asian Fusian')
             | (df['categories_combined']=='Steakhouse')
             | (df['categories_combined']=='Salad')
             | (df['categories_combined']=='Cafes')
             | (df['categories_combined']=='Barbeque')
             | (df['categories_combined']=='Southern')
             | (df['categories_combined']=='Ice Cream & Frozen Yogurt')
             | (df['categories_combined']=='Vietnamese')
             | (df['categories_combined']=='Vegetarian')
             | (df['categories_combined']=='Specialty Food')
             | (df['categories_combined']=='Mediterranean ')
             | (df['categories_combined']=='Local Flavor')
             | (df['categories_combined']=='Indian')             
             | (df['categories_combined']=='Tex-Mex')]    

print('\nDimensions after filtering food categories with over 30k counts:',
      df.shape) #(3731061, 10)
print('======================================================================')

###############################################################################
# Initial EDA of reviews - food with > 30k counts
# Word count for each review
df['review_wordCount'] = df['text_reviews'].apply(lambda x: len(str(x).split()))

# Character count for each review
df['review_charCount'] = df['text_reviews'].apply(lambda x: len(str(x)))

# Length of words and characters in each initial review
f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(15, 10), sharex=True)
f.suptitle('Yelp Reviews: Length of Words and Characters for Each Review',
           fontsize = 25)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=30)
#f.supxlabel('Review Number')
ax1.plot(df['review_wordCount'], color = 'red')
ax1.set_ylabel('Word Count', fontsize=20)
ax2.plot(df['review_charCount'], color = 'blue')
ax2.set_ylabel('Character Count', fontsize=20)

f.savefig('lengthInitialReviews_food30k.png', dpi=my_dpi, 
          bbox_inches='tight')

df = df.drop(['review_wordCount', 'review_charCount'], axis=1)

###############################################################################
# Find language of the reviews to filter only English reviews
# Convert from pandas df to dask df
ddata = dd.from_pandas(df, npartitions=10)

del df

print('Time for cleaning with Langid to find the language of the reviews..')
search_time_start = time.time()
# Start timer for experiment
start_time = datetime.now()
print("%-20s %s" % ("Start Time", start_time))
ddata['language'] = ddata['text_reviews'].apply(langid.classify,
                                                meta=('text_reviews',
                                                      'object')).compute(scheduler='processes')
# End timer for experiment
end_time = datetime.now()
print('Finished cleaning with Langid in:', time.time() - search_time_start)
print("%-20s %s" % ("Start Time", start_time))
print("%-20s %s" % ("End Time", end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

df = ddata.compute()
del ddata

print('======================================================================')

# Generate label for language
df['language'] = df['language'].apply(lambda tuple: tuple[0])

# Number unique language labels
print('Number of tagged languages (estimated):')
print(len(df['language'].unique()))
print('======================================================================')

# Percent of the total dataset in English
print('Percent of data in English (estimated):')
print((sum(df['language']=='en')/len(df))*100) 
print('======================================================================')

# Find reviews with non-English words
df1 = df.loc[df['language'] != 'en']

print('\nNumber of non-English reviews:', df1.shape[0])
print('======================================================================')

del df1

# Subset only English reviews
df = df.loc[df['language'] == 'en']
print('\nNumber of English reviews:', df.shape[0])
print('======================================================================')

df = df.drop(['language'], axis=1)

###############################################################################
# Define class for removing non words before processing words
class cleantext():
    
    def __init__(self, text = 'test'):
        self.text = text
        
    def remove_between_square_brackets(self):
        self.text = re.sub('\[[^]]*\]', '', self.text)
        return self

    def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)
        return self
    
    def replace_contractions(self):
        '''Replace contractions in string of text'''
        self.text = contractions.fix(self.text)
        return self
   
    def remove_special_characters(self, remove_digits=True):
        self.text = re.sub('[^a-zA-z0-9\s]','', self.text)
        return self

    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self
    
    def remove_non_ascii(self):
        '''Remove non-ASCII characters from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = unicodedata.normalize('NFKD',
                                             word).encode('ascii',
                                                          'ignore').decode('utf-8',
                                                                           'ignore')
            new_words.append(new_word)
        self.words = new_words
        return self
    
    def remove_punctuation(self):
        '''Remove punctuation from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        self.words = new_words
        return self

    def join_words(self):
        self.words = ' '.join(self.words)
        return self
    
    def do_all(self, text):
        
        self.text = text
        self = self.remove_numbers()
        self = self.replace_contractions()        
        self = self.get_words()
        self = self.remove_punctuation()
        self = self.remove_non_ascii()
        self = self.remove_special_characters()
        
        return self.words    
    
# Shorter class name for following use
ct = cleantext()

# Define function for using for computations in Dask
def dask_this(df):
    res = df.apply(ct.do_all)
    return res

# Convert from pandas df to dask df
ddata = dd.from_pandas(df, npartitions=10)

del df

print('Time for reviews to be cleaned for non-words...')
search_time_start = time.time()
# Start timer for experiment
start_time = datetime.now()
print("%-20s %s" % ("Start Time", start_time))
ddata['cleanReview'] = ddata['text_reviews'].map_partitions(dask_this).compute(scheduler='processes')

# End timer for experiment
end_time = datetime.now()

print('Finished cleaning reviews in:', time.time() - search_time_start)
print("%-20s %s" % ("Start Time", start_time))
print("%-20s %s" % ("End Time", end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

df = ddata.compute()

del ddata

# Drop original review strings
df = df.drop(['text_reviews'], axis=1)

# Remove comma from tokenize to make one string
df = df.copy()
df['cleanReview'] = df['cleanReview'].apply(lambda x: ','.join(map(str, x)))
df.loc[:,'cleanReview'] =  df['cleanReview'].str.replace(r',', ' ', regex=True)
print('======================================================================')

###############################################################################
# Define class for processing words   
# Some Non UTF8 existed so they had to locate and remove
# Potentially add more common words to this list
stopwords_list = stopwords.words('english')
stopwords_list.extend(('thing','eat'))

class cleantext1():
    
    def __init__(self, text = 'test'):
        self.text = text

    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self

    def to_lowercase(self):
        '''Convert all characters to lowercase from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
        self.words = new_words
        return self
    
    def remove_stopwords(self):
        '''Remove stop words from list of tokenized words'''
        new_words = []
        for word in self.words:
            if word not in stopwords_list:
                new_words.append(word)
        self.words = new_words
        return self

    def lemmatize_words(self):
        '''Lemmatize words in list of tokenized words'''
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in self.words:
            lemma = lemmatizer.lemmatize(word)
            lemmas.append(lemma)
        self.words = lemmas
        return self

    def join_words(self):
        self.words = ' '.join(self.words)
        return self
    
    def do_all(self, text):
        
        self.text = text
        self = self.get_words()
        self = self.to_lowercase()
        self = self.remove_stopwords()
        self = self.lemmatize_words()
        
        return self.words

# Shorter class name for following use
ct = cleantext1()

# Define function for using for computations in Dask
def dask_this(df):
    res = df.apply(ct.do_all)
    return res

ddata = dd.from_pandas(df, npartitions=10)

del df

print('Time for reviews to be cleaned for stopwords and lemma...')
search_time_start = time.time()
# Start timer for experiment
start_time = datetime.now()
print("%-20s %s" % ("Start Time", start_time))

ddata['cleanReview1'] = ddata['cleanReview'].map_partitions(dask_this).compute(scheduler='processes')

# End timer for experiment
end_time = datetime.now()

print('Finished cleaning reviews in:', time.time() - search_time_start)
print("%-20s %s" % ("Start Time", start_time))
print("%-20s %s" % ("End Time", end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

df = ddata.compute()

del ddata

# Remove comma from tokenize to make one string
df = df.copy()
df['cleanReview'] = df['cleanReview1'].apply(lambda x: ','.join(map(str, x)))
df.loc[:,'cleanReview'] =  df['cleanReview'].str.replace(r',', ' ', regex=True)

###############################################################################
# EDA of cleaned reviews
# Word count for each review
df['cleanReview_wordCount'] = df['cleanReview'].apply(lambda x: len(str(x).split()))

# Character count for each review
df['cleanReview_charCount'] = df['cleanReview'].apply(lambda x: len(str(x)))

# Length of words and characters in each cleaned review
f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(15, 10), sharex=True)
f.suptitle('Yelp Reviews: Length of Words and Characters for Cleaned Review',
           fontsize = 25)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=30)
#f.supxlabel('Review Number')
ax1.plot(df['cleanReview_wordCount'], color = 'red')
ax1.set_ylabel('Word Count', fontsize=20)
ax2.plot(df['cleanReview_charCount'], color = 'blue')
ax2.set_ylabel('Character Count', fontsize=20)

f.savefig('lengthCleanedReviews_food30k.png', dpi=my_dpi, 
          bbox_inches='tight')

###############################################################################
# Subset by review stars into different sets
df1 = df.loc[(df['stars_reviews'] == 1) | (df['stars_reviews'] == 2)]
#df1 = df[df.stars_reviews == 3]

df2 = df[df.stars_reviews == 5]
#df2 = df[df.stars_reviews == 4]


# Word cloud visualization
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = ' '.join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    # Change name depending on subset
    plt.savefig('wordCloud_5star.png', bbox_inches='tight',
                dpi=my_dpi * 10)
    plt.axis('off')  
    plt.show()

# Subset string for word cloud  
df1_clean = df1['cleanReview']
df2_clean = df2['cleanReview']
    
print('Higher Rated Reviews')
wordcloud_draw(df2_clean,'white')

print('Lower Rated Reviews')
wordcloud_draw(df1_clean, 'black')

del df1_clean, df2_clean
 
###############################################################################
# Find top frequency of words in reviews                                                                       
df1_clean = df1['cleanReview1']
df2_clean = df['cleanReview1']
#df_clean = df['cleanReview1']

# Defines a function for finding a list of words in the reviews
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

ratings_lower_words = get_all_words(df1_clean)
ratings_higher_words = get_all_words(df2_clean)
                              
# Find 100 most common words in lowest stars
freq_dist_lower = FreqDist(ratings_lower_words)
print(freq_dist_lower.most_common(100))

list1 = freq_dist_lower.most_common(100)
with open('topWords_3starReviews.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    writer.writerows(list1)
    
print('======================================================================')

# Find 100 most common words in 5 star reviews 
freq_dist_higher = FreqDist(ratings_higher_words)
print(freq_dist_higher.most_common(100))

# Write list to csv
list0 = freq_dist_higher.most_common(100)
#with open('topWords_allReviews.csv','w') as f:
with open('topWords_allReviews.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    writer.writerows(list0)

print('======================================================================')

# Find words that are in both 5 and 1,2 star reviews
df1 = pd.DataFrame (list0, columns = ['word', 'count'])
df1 = df1[['word']]

df2 = pd.DataFrame (list1, columns = ['word', 'count'])
df2 = df2[['word']]  

df1 = df1.merge(df2, on='word')
df1.to_csv('wordsSimilar_34reviews.csv', index=False)      


del df1, df2, df1_clean, df2_clean

###############################################################################
# All reviews
df1 = pd.read_csv('topWords_allReviews.csv')
df1.columns
df1.head()
df1 = df1[['word']]

df2 = pd.read_csv('topWords_12starReviews.csv')
df2 = df2[['word']]  
df2.rename(columns={'word': 'word1'}, inplace=True)

df3 = pd.read_csv('topWords_3starReviews.csv')
df3 = df3[['word']]  
df3.rename(columns={'word': 'word2'}, inplace=True)

df4 = pd.read_csv('topWords_4starReviews.csv')
df4 = df4[['word']]  
df4.rename(columns={'word': 'word3'}, inplace=True)

df5 = pd.read_csv('topWords_5starReviews.csv')
df5 = df5[['word']]  
df5.rename(columns={'word': 'word4'}, inplace=True)

df6 = df1.merge(df2, how='outer', left_on='word', right_on='word1') #120
df7 = df6.merge(df3, how='outer', left_on='word', right_on='word2') #120
df8 = df7.merge(df4, how='outer', left_on='word', right_on='word3') #120
df9 = df8.merge(df5, how='outer', left_on='word', right_on='word4') #120

df10 = df9.dropna(how = 'any')
df11 = df10.iloc[:,[0]]


df11.to_csv('wordsSimilar_allreviewsMerge.csv', index=False)      

df12 = df11.values.tolist()
df12

['food'],
 ['place'],
 
 ['would'],
 ['one'],
 ['get'],
 ['back'],
 ['go'],

 ['restaurant'],
 ['order'],
 ['also'],
 ['ordered'],
 ['chicken'],
 ['menu'],
 
 ['come'],

 ['even'],
 ['try'],
 ['pizza'],
 ['could'],
 ['table'],
 ['came'],
 ['little'],
 ['make'],
 ['drink'],
 ['sauce'],
 ['staff'],
 ['first'],
 ['much'],
 ['dish'],
 ['wait'],
 ['price'],
 ['people'],
 ['meal'],
 ['went'],
 ['cheese'],
 ['flavor'],
 ['better'],
 ['made'],
 ['experience'],
 ['friend'],
 ['salad'],
 ['two'],
 ['bar'],
 ['know'],
 ['burger'],
 ['day'],
 ['way'],
 ['say'],
 ['night'],
 ['want'],
 ['going'],
 ['take'],

 ['sandwich'],
 ['dinner'],
 ['around'],
 ['sure']]
###############################################################################
# Write processed reviews + variables for later use
pd.to_pickle(df, './YelpReviews_NLP.pkl')

# Save model
Pkl_Filename = 'YelpReviews_NLP1.pickle' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(df, file, protocol=3)

df1.to_parquet('YelpReviews_NLP.parquet')

import tables

df1 = df.iloc[0:100]

df1.to_parquet('YelpReviews_NLP2.parquet')

df1.to_hdf('YelpReviews_NLP2.h5', key='df1', mode='w')
# Save model
Pkl_Filename = 'YelpReviews_NLP2.pickle' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(df1, file, protocol=2)

df1.to_hdf('YelpReviews_NLP2.h5', key='df1', mode='w')


# conda install -c conda-forge pytables
import pytables
# Change protocol number
pickle.HIGHEST_PROTOCOL = 4
df.to_hdf('YelpReviews_NLP1.hdf', 'df')

data = pd.read_hdf(YelpReviews_NLP1.hdf) 

pd.to_pickle(df, './YelpReviews_NLP1.pkl')
###############################################################################

# Read data
df = pd.read_pickle('./YelpReviews_NLP.pkl')
print('Number of rows and columns:', df.shape)
df.head()

df = df[['cleanReview1', 'stars_reviews']]

#df = df1
df = df.copy()
df[['review_rank']] = df[['stars_reviews']]

df['review_rank'].mask(df['review_rank'] == 1, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 2, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 5, 1, inplace=True)

df1 = df[df.review_rank==1]
df1 = df1.sample(n=770743)

df2 = df[df.review_rank==0]

df3 = pd.concat([df1, df2])

from sklearn.utils import shuffle
df3 = shuffle(df3)

df3.head()

del df, df1, df2

data = df3.cleanReview1
target = df3.review_rank

del df3
#del ata, df1, df2, df4, X_train, X_test, y_train, y_test

###############################################################################
df.head()
from textblob import TextBlob

df['polarity'] = df['cleanReview'].apply(lambda x: TextBlob(x).sentiment[0])
print(df.isna().sum())

print(df['polarity'].describe())
# =============================================================================
# count    3.731061e+06
# mean     2.522549e-01
# std      2.281180e-01
# min     -1.000000e+00
# 25%      1.250000e-01
# 50%      2.584499e-01
# 75%      3.916667e-01
# max      1.000000e+00
# 
# =============================================================================
print(df[['polarity']].mean()) #0.252255

#df[['stars_reviews', 'polarity','cleanReview']].head()

#df[['stars_reviews', 'polarity','cleanReview']].tail()
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

df = df.join(pd.DataFrame(sentiment))
df.rename(columns={0: 'sentiment'}, inplace=True)

#df = df.drop(['sentiment'], axis=1)
# =============================================================================
# import itertools
# df['sentiment'] = list(itertools.chain(*sentiment))
# 
# #can only concatenate list (not "str") to list
# df['sentiment']=sum(sentiment,[])
# =============================================================================

df[['stars_reviews', 'sentiment']].value_counts()

# =============================================================================
# stars_reviews  sentiment        
# 5.0            Slightly Positive    658966
#                Positive             595783
# 4.0            Slightly Positive    474073
#                Slightly Negative    256261
# 1.0            Negative             231245
# 4.0            Positive             223256
# 5.0            Slightly Negative    221781
# 3.0            Slightly Negative    207974
#                Slightly Positive    181557
# 2.0            Slightly Negative    171426
# 1.0            Slightly Negative    153952
# 2.0            Negative              90320
#                Slightly Positive     68754
# 3.0            Positive              47295
#                Negative              44097
# 1.0            Slightly Positive     33357
# 4.0            Negative              26276
# 5.0            Negative              22999
# 2.0            Positive              14140
# 1.0            Positive               7549
# =============================================================================

df[['sentiment']].value_counts()
# =============================================================================
# sentiment        
# Slightly Positive    1416707
# Slightly Negative    1011394
# Positive              888023
# Negative              414937
# dtype: int64
# 
# =============================================================================
df.columns

# =============================================================================
# df['sentiment'] = sentiment
# sentiment[0:10] # list
# g = df1.groupby('sentiment')['stars_reviews'].count() \
#                              .reset_index(name='count') \
#                              .sort_values(['count'], ascending=False) 
# 
# g
# =============================================================================
                             
df[['stars_reviews']].value_counts()
# =============================================================================
# 5.0              1499529
# 4.0               979866
# 3.0               480923
# 1.0               426103
# 2.0               344640
# =============================================================================
df1 = df

df = df.copy()
df[['review_rank']] = df[['stars_reviews']]

df['review_rank'].mask(df['review_rank'] == 1, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 2, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 3, 1, inplace=True)
df['review_rank'].mask(df['review_rank'] == 4, 1, inplace=True)
df['review_rank'].mask(df['review_rank'] == 5, 2, inplace=True)

df.head()

df[['review_rank']].value_counts() 
# =============================================================================
# 2.0            1499529
# 1.0            1460789
# 0.0             770743
# =============================================================================
df.head()

# =============================================================================
# # 1,2 vs 4,5 example
# simple binary classifcation problem with positive i.e, high polarity(stars= 4 and 5) 
# and negative(stars= 1 and 2) labels I have converted stars column as binary column.
#  We can keep neutral class as stars =3 but right now i am not dealing with that for sake of simplicity.
#df=df[df.stars_reviews!=3]
# df['labels'= df['stars_reviews'].apply(lambda x: 1 if x > 3  else 0)
# df=df.drop("'stars_reviews',axis=1)
# =============================================================================
df = df1
df = df.copy()
df[['review_rank']] = df[['stars_reviews']]

df['review_rank'].mask(df['review_rank'] == 1, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 2, 0, inplace=True)
df['review_rank'].mask(df['review_rank'] == 5, 1, inplace=True)

df1 = df[df.review_rank==1]
df1 = df1.sample(n=770743)

df2 = df[df.review_rank==0]

df3 = pd.concat([df1, df2])

from sklearn.utils import shuffle
df3 = shuffle(df3)

df3.head()
df1 = df3[['cleanReview', 'stars_reviews']]
df1.to_parquet('YelpReviews_NLP_125stars.parquet')

# Add "" to everything 
df1 = df.copy()
df1["cleanReview"]= [[f'"{j}"' for j in i] for i in df1["cleanReview1"]]
df1.head()

df1 = df1[['cleanReview', 'stars_reviews']]


# Write processed reviews + variables for later use
pd.to_pickle(df3, './YelpReviews_NLP_125star.pkl')

# Sample 5 star to match count
#df_rank = df[(df.review_rank==0) | (df.review_rank==1)]


data = df3.cleanReview1
target = df3.review_rank


data = df3.cleanReview1.tolist()

df4

df3['tokenized'] = data
df3.columns
df3.head()
df1 = df3[['cleanReview', 'stars_reviews']]
df1.to_parquet('YelpReviews_NLP_125stars.parquet')

###############################################################################
from sklearn.model_selection import train_test_split
#, stratify=target
X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                    test_size=0.2, stratify=target)

train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

import seaborn as sns
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
sns.barplot(x=train_counts.index,y=train_counts.values, ax=ax1)
sns.barplot(x=test_counts.index,y=test_counts, ax=ax2)
ax1.set_title('TRAIN DATA: Num Reviews by Star Rating');
ax2.set_title('TEST DATA: Num Reviews by Star Rating');

y_train.shape, y_test.shape

print('Number of Reviews in Target Groups:')
print(y_test.value_counts())

#Gensim
import gensim
from gensim.models import Word2Vec

import multiprocessing
cores = multiprocessing.cpu_count()
cores #16

# Generate Word2Vec Word Embeddings
dim = 50 #Dimensionality of word vectors

print('Generate Word Vectors:', end=' ')
start = time.time()
model = Word2Vec(X_train, size=dim, window=5, min_count=1, workers=cores-1)
model
end = time.time()
print(round(end-start,2), 'seconds')
# Save model
#model.wv.save_word2vec_format('model.bin')
# Save as .txt
#model.wv.save_word2vec_format('model.txt', binary=False)

# load model
#model = Word2Vec.load('model.bin')

# =============================================================================
# # PCA and scatter plot
# from sklearn.decomposition import PCA
# from matplotlib import pyplot
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
# 
# =============================================================================
# Training of model
print('Train Model:', end=' ')
start = time.time()
model.train(data, total_examples=model.corpus_count, epochs=30)
end = time.time()
print(round(end-start,2), 'seconds')
# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Save model
#model.wv.save_word2vec_format('model_dim50_win1_mincount1_30epochs.bin')

# Make model more memory efficient since finished training
model.init_sims(replace=True)
# Save embedded word vector space
wv = model.wv
print(len(wv.vocab),'unique words in the dataset.')


# =============================================================================
# Generate Word Vectors: 388.77 seconds
# 325185 unique words in the dataset.
# ==
# =============================================================================
# 437.54 seconds
# 325185 unique words in the dataset.
# =============================================================================

###############################################################################
# Train models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

results = []

# =============================================================================
# # Run 1
# param_grid_cb = [{'cb__max_depth': [3],
#                    'cb__n_estimators': [100],
#                    'cb__learning_rate': [0.01, 0.001],
#                    'cb__l2_leaf_reg': [0.1, 1] }]
# =============================================================================

# Run 2
param_grid_cb = [{'cb__max_depth': [3, 5, 7, 10],
                   'cb__n_estimators': [100, 200, 300, 500],
                   'cb__learning_rate': [0.01, 0.001, 0.0001],
                   'cb__l2_leaf_reg': [0.1, 0.5, 1],
                   'cb__min_data_in_leaf': [5, 10, 15] }]

#Creating Mean Word Embeddings using Mean Embedding Vectorizer class
class W2vVectorizer(object):
    """
    This class is used to provide mean word vectors for review documents. 
    This is done in the transform function which is used to generate mean vectors in model pipelines.
    The class has both fit and transform functions so that it may be used in an sklearn Pipeline.
    """
    
    def __init__(self, w2v):
        self.w2v = w2v
        
        #If using GloVe the model is in a dictionary format
        if isinstance(w2v, dict):
            if len(w2v) == 0:
                self.dimensions = 0
            else:
                self.dimensions = len(w2v[next(iter(w2v))])
        #Otherwise, using gensim keyed vector
        else:
            self.dimensions = w2v.vector_size
    
    # Need to implement a fit method as required for sklearn Pipeline.
    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        This function generates a w2v vector for a set of tokens. This is done by taking 
        the mean of each token in the review.
        """
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dimensions)], axis=0) for words in X])


cb  = Pipeline([("W2vVectorizer", W2vVectorizer(wv)), 
                ("cb", CatBoostClassifier(task_type='GPU', 
                                          early_stopping_rounds=10,
                                          random_state=seed_value,))])

models = [("Catboost", cb, param_grid_cb)]

#from joblib import parallel_backend
def get_gridsearch_result(name, estimator, param_grid, X_train, X_test, y_train, y_test, cv=5, scoring='accuracy'):
    """
    This function fits a GridSearchCV model and populates a dictionary containing the best model and accuracy results.
    
    INPUTS:
    name       = The name of the gridsearch model desired. It will be used as a key in the returned dictionary object.
    estimator  = The model which will be passed into GridSearchCV. It can be a pipeline model or a base model.
    param_grid = The parameter grid to be used by GridSearchCV.
    X_train, X_test, y_train, y_test = train test split of data(X) and target(y).
    cv         = Number of cross validations to perform.
    
    RETURN:
    Dictionary containing the fitted GridSearchCV model as well as summary metrics.
    Dictionary keys are: name, model, model params, accuracy train, accuracy test, 
                         custom accuracy train, custom accuracy test.
    """
    grid_clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                            scoring=scoring, n_jobs=7)
    grid_clf.fit(X_train,y_train)
    #with parallel_backend('threading', n_jobs=-1): 
    #   grid_clf.fit(X_train,y_train)

    result = {}
    
    result['name'] = name
    result['model'] = grid_clf.best_estimator_
    result['model params'] = grid_clf.best_params_

    y_pred_train = grid_clf.predict(X_train)
    y_pred_test  = grid_clf.predict(X_test)

    result['accuracy train']  = round(accuracy_score(y_train,y_pred_train),4)
    result['accuracy test']   = round(accuracy_score(y_test,y_pred_test),4)
    result['custom accuracy train'] = round(custom_accuracy(y_train,y_pred_train),4)
    result['custom accuracy test']  = round(custom_accuracy(y_test,y_pred_test),4)

    return result

def gridsearch_all_models(models, X_train, X_test, y_train, y_test, cv=5, scoring='accuracy'):
    """
    This function will perform a grisearch on a list of models, output the time taken, 
    and return a list of results dictionaries for each gridsearch model.
    
    INPUTS:
    models = List of tuples in the form (name, model, param_grid)
        name  = text name of the model
        model = model of pipeline model
        param_grid = parameters to be used for gridsearch
    X_train, X_test, y_train, y_test = train test split of data(X) and target(y).
    
    RETURNS:
    List of dictionaries containing gridsearch model, selected parameters, and accuracy scores.
    """
    
    print("GRIDSEARCH AND SCORE ALL MODELS:")
    start = time.time()
    results = []

    for name, model, param_grid in models:
        start_model = time.time()
        print("  ", name, end='')
        results.append(get_gridsearch_result(name=name,
                                             estimator=model,
                                             param_grid=param_grid,
                                             X_train=X_train,
                                             X_test=X_test, 
                                             y_train=y_train,
                                             y_test=y_test,
                                             cv=cv,
                                             scoring=scoring))

        end_model = time.time()
        print(":\t time", time.strftime('%H:%M:%S', time.gmtime(end_model-start_model)))

    end = time.time()
    print("TOTAL TIME:", time.strftime('%H:%M:%S', time.gmtime(end-start)))
    return results

def custom_accuracy(y_true, y_pred, threshold=1, credit_given=0.5):
    """
    If y_pred is off by threshold or less, give partial credit according to credit_given.
    INPUTS:
    y_true       = True label.
    y_pred       = Predicted label.
    threshold    = Threshold for giving credit to inaccurate predictions. (=1 gives credit to predictions off by 1)
    credit_given = Partial credit amount for close but inaccurate predictions. (=0.5 give 50% credit)
    """
    predicted_correct = sum((y_true-y_pred)==0)
    predicted_off = sum(abs(y_true-y_pred)<=threshold) - predicted_correct
    custom_accuracy = (predicted_correct + credit_given*predicted_off)/len(y_true)
    return custom_accuracy

my_scoring_function = make_scorer(custom_accuracy)

results.extend(gridsearch_all_models(models, X_train, X_test, y_train, y_test,
                                     scoring=my_scoring_function))

model_results_df = pd.DataFrame(results)
print(model_results_df)

model_results_df.to_csv('catboost_125reviews.csv', index=False)
###############################################################################
#PARAMETER GRIDS FOR GRIDSEARCH
param_grid_rf  = [{'Random Forest__criterion': ['entropy'],
                   'Random Forest__max_depth': [6, 7],
                   'Random Forest__n_estimators': [100] }]

param_grid_svc = [{'Support Vector Machine__C': [1],
                   'Support Vector Machine__gamma': ['auto'] }]

param_grid_lr  = [{'Logistic Regression__solver': ['sag','lbfgs'],
                   'Logistic Regression__penalty': ['l2'],
                   'Logistic Regression__C': [.01, .1, 1],
                   'Logistic Regression__max_iter': [1000]}]

param_grid_xgb = [{'XGBoost Model__max_depth': [3],
                   'XGBoost Model__n_estimators': [100],
                   'XGBoost Model__learning_rate': [0.1, 0.2],
                   'XGBoost Model__gamma': [0, 1] }]

#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

#Creating Mean Word Embeddings using Mean Embedding Vectorizer class
class W2vVectorizer(object):
    """
    This class is used to provide mean word vectors for review documents. 
    This is done in the transform function which is used to generate mean vectors in model pipelines.
    The class has both fit and transform functions so that it may be used in an sklearn Pipeline.
    """
    
    def __init__(self, w2v):
        self.w2v = w2v
        
        #If using GloVe the model is in a dictionary format
        if isinstance(w2v, dict):
            if len(w2v) == 0:
                self.dimensions = 0
            else:
                self.dimensions = len(w2v[next(iter(w2v))])
        #Otherwise, using gensim keyed vector
        else:
            self.dimensions = w2v.vector_size
    
    # Need to implement a fit method as required for sklearn Pipeline.
    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        This function generates a w2v vector for a set of tokens. This is done by taking 
        the mean of each token in the review.
        """
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dimensions)], axis=0) for words in X])

   ################################ 
    def do_all(self, model):
        
        self.w2v = model
        self = self.fit()
        self = self.transform()

        
        return self.model
    
# Shorter class name for following use
ct = W2vVectorizer()

# Define function for using for computations in Dask
def dask_this(df):
    res = df.apply(ct.do_all)
    return res

ddata = dd.from_pandas(wv, npartitions=10)

####################################################################
#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

#XGBoost
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from xgboost import XGBClassifier

#CREATE PIPELINE MODELS AND MODELS LIST
rf  = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(wv)),
                ("Random Forest", RandomForestClassifier(max_features='auto'))])
svc = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(wv)),
                ("Support Vector Machine", SVC())])
lr  = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(wv)),
                ("Logistic Regression", LogisticRegression())]) #multi_class='multinomial'
xgb = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(wv)),
                ("XGBoost Model", XGBClassifier())])

models = [("Random Forest", rf, param_grid_rf)]

models = [("Logistic Regression", lr, param_grid_lr),
          ("XGBoost Model", xgb, param_grid_xgb)]

models = [("Random Forest", rf, param_grid_rf),
          ("Support Vector", svc, param_grid_svc),
          ("Logistic Regression", lr, param_grid_lr),
          ("XGBoost Model", xgb, param_grid_xgb)]


# =============================================================================
# GRIDSEARCH AND SCORE ALL MODELS:
#    Random Forest:	 time 00:31:59
# TOTAL TIME: 00:31:59
# =============================================================================

# =============================================================================
# GRIDSEARCH AND SCORE ALL MODELS:
#    Random Forest:	 time 01:11:31
# TOTAL TIME: 01:11:31
# =============================================================================

###############################################################################
# =============================================================================
# Predict Ratings using GloVe Word Vectors and several models
# - Random Forest
# - Support Vector Machine
# - Logistic Regression
# - XGBoost
# =============================================================================

def load_GloVe_vectors(file, vocab):
    """
    This function will load Global Vectors for words in vocab from the specified GloVe file.
    
    INPUT:
    file  = The path/filename of the file containing GloVe information.
    vocab = The list of words that will be loaded from the GloVe file.
    """
    glove = {}
    with open(file, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                glove[word] = vector
    return glove

# Set path
path = r'D:\Yelp_Reviews\Data\glove.6B'
os.chdir(path)

glove = load_GloVe_vectors(file='glove.6B.50d.txt', vocab=wv.vocab.keys())
#glove = load_GloVe_vectors(file='glove.840B.300d.txt', vocab=wv.vocab.keys())

# Load Google’s Word2Vec Embedding
from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# Set path
path = r'D:\Yelp_Reviews\EDA\NLP'
os.chdir(path)

#CREATE PIPELINE MODELS AND MODELS LIST
rf  = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("Random Forest", RandomForestClassifier(max_features='auto'))])
svc = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("Support Vector Machine", SVC())])
lr  = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("Logistic Regression", LogisticRegression(multi_class='multinomial'))])
xgb = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("XGBoost Model", XGBClassifier())])

glove_models = [("GloVe- Random Forest", rf, param_grid_rf)]


glove_models = [("GloVe- Random Forest", rf, param_grid_rf),
                ("GloVe- Support Vector", svc, param_grid_svc),
                ("GloVe- Logistic Regression", lr, param_grid_lr),
                ("GloVe- XGBoost Model", xgb, param_grid_xgb)]

results.extend(gridsearch_all_models(glove_models, X_train, X_test, y_train,
                                     y_test, scoring=my_scoring_function))

# =============================================================================
# GRIDSEARCH AND SCORE ALL MODELS:
#    GloVe- Random Forest:	 time 01:18:46
# TOTAL TIME: 01:18:46
# 
# =============================================================================

# =============================================================================
# model_results_df = pd.DataFrame(results)
# model_results_df
# Out[18]: 
#                    name                                              model  \
# 0  GloVe- Random Forest  (<__main__.W2vVectorizer object at 0x000002225...   
# 
#                                         model params  accuracy train  \
# 0  {'Random Forest__criterion': 'entropy', 'Rando...          0.8056   
# 
#    accuracy test  custom accuracy train  custom accuracy test  
# 0         0.8023                 0.9028                0.9012  
# =============================================================================
# catboost
#TOTAL TIME: 02:46:56

model_results_df = pd.DataFrame(results)
model_results_df

model_results_df.to_csv('cb_125reviews.csv', index=False)

# Run sequentially

results.extend(gridsearch_all_models(models, X_train, X_test, y_train, y_test,
                                     scoring=my_scoring_function))


lr  = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("Logistic Regression", LogisticRegression())])
xgb = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ("XGBoost Model", XGBClassifier())])

glove_models = [("GloVe- Logistic Regression", lr, param_grid_lr),
                ("GloVe- XGBoost Model", xgb, param_grid_xgb)]

results.extend(gridsearch_all_models(glove_models, X_train, X_test, y_train,
                                     y_test, scoring=my_scoring_function))

model_results_df = pd.DataFrame(results)
model_results_df

model_results_df.to_csv('mlAlgs_125reviews.csv', index=False)
###############################################################################
#XGBoost
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from xgboost import XGBClassifier

#Keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, GlobalMaxPool1D
from keras.preprocessing import text, sequence

embedding_size = 128
input_ = Input(shape=(100,))
x = Embedding(20000, embedding_size)(input_)
x = LSTM(25, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.5)(x)
# There are 5 different possible rating levels, therefore using 5 neurons in output layer
x = Dense(5, activation='softmax')(x)

keras_model = Model(inputs=input_, outputs=x)

keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
keras_model.summary()

param_grid_km  = [{'Keras LSTM Model__epochs': [2],
                   'Keras LSTM Model__batch_size': [32] }]

from keras.preprocessing import text, sequence
from sklearn.pipeline import Pipeline

class KerasTokenizer(object):
    """
    This class is used to fit text and convert text to sequences for use in a Keras NN Model.
    The class has both fit and transform functions so that it may be used in an sklearn Pipeline.
    num_words = max number of words to keep.
    maxlen  = max length of all sequences.
    """
    def __init__(self, num_words=20000, maxlen=100):
        self.tokenizer = text.Tokenizer(num_words=num_words)
        self.maxlen = maxlen
        
    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        return self
        
    def transform(self, X):
        return sequence.pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.maxlen)
    
class KerasModel(object):
    """
    This class is used to fit and transform a keras model for use in an sklearn Pipeline.
    """
    def __init__(self, model, epochs=3, batch_size=32, validation_split=0.1):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
    def set_params(self, epochs=3, batch_size=32, validation_split=0.1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        return self
    
    def fit(self, X, y):
        y_dummies = pd.get_dummies(y).values
        self.labels = np.array(pd.get_dummies(y).columns)
        self.model.fit(X, y_dummies, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        return self
    
    def transform(self, X):
        return X
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return [self.labels[idx] for idx in y_pred.argmax(axis=1)]
    
    def summary(self):
        self.model.summary()
        
km  = Pipeline([("Keras Tokenizer", KerasTokenizer(num_words=20000, maxlen=100)),
                 ("Keras LSTM Model", KerasModel(keras_model))])

model = [("Keras Model", km, param_grid_km)]

results.extend(gridsearch_all_models(model, X_train, X_test, y_train, y_test, cv=5, scoring=my_scoring_function))

#CHOSEN MODELS
chosen_model_keras = model_results_df.loc[8].model
chosen_model_svc = model_results_df.loc[1].model

###############################################################################
# Bidrectional LSTM Spacy
reviews=yelp_reviews[:300000]
reviews=reviews[reviews.stars!=3]

reviews["labels"]= reviews["stars"].apply(lambda x: 1 if x > 3  else 0)
reviews=reviews.drop("stars",axis=1)


texts = reviews["text"].values
labels = reviews["labels"].values


# 
# ### Converting text into numerical representation i.e Tensors
# Then we can format our text samples and labels into tensors that can be fed into a neural network. 
# Some Preprocessing is needed here.
# Tokenization - We need to break down the sentence into unique words. For eg, "The cow jumped over the moon" will become ["The","cow","jumped","over","moon"]
# 
# To do this, we will rely on Keras utilities keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
MAX_NUM_WORDS=1000 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH=100 # max number of words in a review to use


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# ###  split the data into a training set and a validation set
VALIDATION_SPLIT=0.2

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# ### Preparing the Embedding layer
GLOVE_DIR='../input/glove-global-vectors-for-word-representation/'

import os
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# ### compute  embedding matrix
# At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix
EMBEDDING_DIM = 50 # how big is each word vector

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# ### Define Embedding Layer 
#  We load this embedding matrix into an Embedding layer. 
# Note that we set trainable=False to prevent the weights from being updated during training.

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# ### Training model :)
from keras.layers import Bidirectional, GlobalMaxPool1D,Conv1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.models import Model


inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedded_sequences = embedding_layer(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(2, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128);

# This gives us 93% accuracy. Thats not enough may be we should bring on more data.
# 
# 
# ### Spacy
# 
# spaCy is an industrial-strength natural language processing (NLP) library for Python.
# spaCy's goal is to take recent advancements in natural language processing out of
# research papers and put them in the hands of users to build production software.
# 
# spaCy handles many tasks commonly associated with building an end-to-end natural language processing pipeline:
# 
# 1. Tokenization
# 2. Text normalization, such as lowercasing, stemming/lemmatization
# 3. Part-of-speech tagging
# 4. Syntactic dependency parsing
# 5. Sentence boundary detection
# 6. Named entity recognition and annotation
# 
# Let's take a sample review


import spacy
nlp = spacy.load('en')
sample_review=reviews.text[5]
sample_review




get_ipython().run_cell_magic('time', '', 'parsed_review = nlp(sample_review)\nprint(parsed_review)')


# looks like nothing changed. 
# Let's check againn.
# ### Sentence detection and segmentation
# 


for num, sentence in enumerate(parsed_review.sents):
    print ('Sentence {}:'.format(num + 1))
    print (sentence)
    print ('\n')


# ### Named entity detection


for num, entity in enumerate(parsed_review.ents):
    print ('Entity {}:'.format(num + 1), entity, '-', entity.label_)
    print ('\n')


# ### Part of speech tagging


token_text = [token.orth_ for token in parsed_review]
token_pos = [token.pos_ for token in parsed_review]

parts_of_speech=pd.DataFrame(data=list(zip(token_text, token_pos)),columns=['token_text', 'part_of_speech'])
parts_of_speech.head(10)


# ### Text normalization, like stemming/lemmatization and shape analysis
# 
# The work at this stage attempts to reduce as many different variations of similar words into a single term ( different branches all reduced to single word stem). Therefore if we have "running", "runs" and "run", you would really want these three distinct words to collapse into just the word "run". (However of course you lose granularity of the past, present or future tense).
# 



token_lemma = [token.lemma_ for token in parsed_review]
token_shape = [token.shape_ for token in parsed_review]

text_normalized_DF=pd.DataFrame(list(zip(token_text, token_lemma, token_shape)),
             columns=['token_text', 'token_lemma', 'token_shape'])
text_normalized_DF.head()


# ### Token-level entity analysis



token_entity_type = [token.ent_type_ for token in parsed_review]
token_entity_iob = [token.ent_iob_ for token in parsed_review]

entity_analysis=pd.DataFrame(list(zip(token_text, token_entity_type, token_entity_iob)),
             columns=['token_text', 'entity_type', 'inside_outside_begin'])
entity_analysis.head()


# ### Token  attributes
# 
# such as the relative frequency of tokens, and whether or not a token matches any of these categories
# 
# stopword
# punctuation
# whitespace
# represents a number
# whether or not the token is included in spaCy's default vocabulary?


token_attributes = [(token.orth_,
                     token.prob,
                     token.is_stop,
                     token.is_punct,
                     token.is_space,
                     token.like_num,
                     token.is_oov)
                    for token in parsed_review]

token_attributes = pd.DataFrame(token_attributes,
                  columns=['text',
                           'log_probability',
                           'stop?',
                           'punctuation?',
                           'whitespace?',
                           'number?',
                           'out of vocab.?'])

token_attributes.loc[:, 'stop?':'out of vocab.?'] = (token_attributes.loc[:, 'stop?':'out of vocab.?']
                                       .applymap(lambda x: u'Yes' if x else u''))
                                               
token_attributes.head()






res = pd.concat([df.assign(name=i) for i in names], ignore_index=True)

data = pd.DataFrame(data)
data.head()
data.rename(columns={0: 'tokenized'}, inplace=True)

df4 = pd.concat([df1, data], axis=1)
df4.columns
df4.rename(columns={0: 'tokenized'}, inplace=True)
df4.head()


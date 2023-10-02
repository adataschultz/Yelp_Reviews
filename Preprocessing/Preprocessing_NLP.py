# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import warnings
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import FreqDist
import matplotlib.pyplot as plt
import dask.dataframe as dd
import langid
import time 
import re
import contractions
import string
import unicodedata
from wordcloud import WordCloud, STOPWORDS
import csv
from textblob import TextBlob
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
my_dpi = 96

# Set seed
seed_value = 42
os.environ['Yelp_Review_NLP'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Download word sets for cleaning
nlp = spacy.load('en_core_web_lg')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

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

print('Number of rows and columns:', df.shape)

# Filtered data to the categories with over 30k counts and the states with the seven highest counts of reviews.
v = df.categories_combined.value_counts()
df = df[df.categories_combined.isin(v.index[v.gt(30000)])]

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

print('- Dimensions after filtering food categories with over 30k counts:',
      df.shape)

df1 = df['state'].value_counts().index[:7]
df = df[df['state'].isin(df1)]

del df1

print('- Dimensions after filtering US states with the 7 highest count of reviews:',
      df.shape)

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
f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(11,7), sharex=True)
f.suptitle('Restaurants > 30k Counts in Top 7 States: Length of Words and Characters for Each Review',
           fontsize=15)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=15)
ax1.plot(df['review_wordCount'], color='red')
ax1.set_ylabel('Word Count', fontsize=15)
ax2.plot(df['review_charCount'], color='blue')
ax2.set_ylabel('Character Count', fontsize=15)
f.savefig('lengthInitialReviews_food30k.png', dpi=my_dpi*10, 
          bbox_inches='tight')

df = df.drop(['review_wordCount', 'review_charCount'], axis=1)

###############################################################################
# Find language of the reviews to filter only English reviews
# Convert from pandas df to dask df
ddata = dd.from_pandas(df, npartitions=10)

del df

print('Time for cleaning with Langid to find the language of the reviews..')
search_time_start = time.time()
ddata['language'] = ddata['text_reviews'].apply(langid.classify,
                                                meta=('text_reviews',
                                                      'object')).compute(scheduler='processes')
# End timer for experiment
print('Finished cleaning with Langid in:', time.time() - search_time_start)

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
print((sum(df['language'] == 'en') / len(df)) * 100) 
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
    
    def __init__(self, text='test'):
        self.text = text
        
    def remove_between_square_brackets(self):
        self.text = re.sub('\[[^]]*\]', '', self.text)
        return self

    def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)
        return self
   
    def remove_special_characters(self, remove_digits=True):
        self.text = re.sub('[^a-zA-z0-9\s]','', self.text)
        return self

    def replace_contractions(self):
        """Replace contractions in string of text"""
        self.text = contractions.fix(self.text)
        return self
    
    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self
    
    def remove_non_ascii(self):
        """Remove non-ASCII characters from list of tokenized words"""
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
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in self.words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        self.words = new_words
        return self

    def to_lowercase(self):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
        self.words = new_words
        return self

    def join_words(self):
        self.words = ' '.join(self.words)
        return self

    def do_all(self, text):

        self.text = text
        self = self.remove_numbers()
        self = self.remove_special_characters()
        self = self.replace_contractions()
        self = self.get_words()
        self = self.remove_non_ascii()
        self = self.remove_punctuation()
        self = self.to_lowercase()

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
ddata['cleanReview'] = ddata['text_reviews'].map_partitions(dask_this).compute(scheduler='processes')
print('Finished cleaning reviews for non-words in:',
      time.time() - search_time_start)

df = ddata.compute()

del ddata

# Drop original review strings
df = df.drop(['text_reviews'], axis=1)

# Remove comma from tokenize to make one string
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
    
    def __init__(self, text='test'):
        self.text = text

    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self
    
    def remove_stopwords(self):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in self.words:
            if word not in stopwords_list:
                new_words.append(word)
        self.words = new_words
        return self

    def lemmatize_words(self):
        """Lemmatize words in list of tokenized words"""
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
ddata['cleanReview1'] = ddata['cleanReview'].map_partitions(dask_this).compute(scheduler='processes')
print('Finished cleaning reviews in:', time.time() - search_time_start)

df = ddata.compute()

del ddata

df['cleanReview1'] = df['cleanReview1'].apply(lambda x: ','.join(map(str, x)))
df[:,'cleanReview1'] =  df['cleanReview1'].str.replace(r',', ' ', regex=True)

###############################################################################
# EDA of cleaned reviews
# Word count for each review
df['cleanReview_wordCount'] = df['cleanReview'].apply(lambda x: len(str(x).split()))

# Character count for each review
df['cleanReview_charCount'] = df['cleanReview'].apply(lambda x: len(str(x)))

# Length of words and characters in each cleaned review
f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(15,10), sharex=True)
f.suptitle('Yelp Reviews: Length of Words and Characters for Cleaned Review',
           fontsize=25)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=30)
ax1.plot(df['cleanReview_wordCount'], color='red')
ax1.set_ylabel('Word Count', fontsize=20)
ax2.plot(df['cleanReview_charCount'], color='blue')
ax2.set_ylabel('Character Count', fontsize=20)
f.savefig('lengthCleanedReviews_food30k.png', dpi=my_dpi*10, 
          bbox_inches='tight')

###############################################################################
# Length of words before/after removing non-words and stopwords/lemmatization
df['cleanReview_wordCount'] = df['cleanReview'].apply(lambda x: len(str(x).split()))
df['cleanReview_wordCount1'] = df['cleanReview1'].apply(lambda x: len(str(x).split()))

f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(11,7), sharex=True, sharey=True)
f.suptitle('Length of Words After Removing Non-Words and Stopwords/Lemmatization',
           fontsize=15)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=15)
ax1.plot(df['cleanReview_wordCount'], color='red')
ax1.set_ylabel('Word Count', fontsize=15)
ax2.plot(df['cleanReview_wordCount1'], color='blue');

df = df.drop(['cleanReview_wordCount', 'cleanReview_wordCount1'], axis=1)

###############################################################################
# length of the characters before/after removing non-words and stopwords/lemmatization.
df['cleanReview_charCount'] = df['cleanReview'].apply(lambda x: len(str(x)))
df['cleanReview_charCount1'] = df['cleanReview1'].apply(lambda x: len(str(x)))

f, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(11,7), sharex=True, sharey=True)
f.suptitle('Length of Characters After Removing Non-Words and Stopwords/Lemmatization',
           fontsize=15)
f.text(0.5, 0.04, 'Review Number', ha='center', fontsize=15)
ax1.plot(df['cleanReview_charCount'], color='red')
ax1.set_ylabel('Character Count', fontsize=15)
ax2.plot(df['cleanReview_charCount1'], color='blue');

###############################################################################
# Subset by review stars into different sets
df1 = df.loc[(df['stars_reviews'] == 1) | (df['stars_reviews'] == 2)]
df2 = df[df.stars_reviews == 5]

# Subset string for word cloud  
df1_clean = df1['cleanReview']
df2_clean = df2['cleanReview']

# Word cloud visualization
def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    cleaned_word = ' '.join([word for word in words.split()
                            if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=1500,
                          height=1000).generate(cleaned_word)
    plt.figure(1, figsize=(10,10))
    plt.imshow(wordcloud)
    # Change name depending on subset
    plt.savefig('wordCloud_5star.png', bbox_inches='tight',
                dpi=my_dpi*10)
    plt.axis('off')  
    plt.show()
    
print('- Higher Rated Reviews')
wordcloud_draw(df2_clean,'white')

print('- Lower Rated Reviews')
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
with open('topWords_5starReviews.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    writer.writerows(list1)
    
print('======================================================================')

# Find 100 most common words in 5 star reviews 
freq_dist_higher = FreqDist(ratings_higher_words)
print(freq_dist_higher.most_common(100))

# Write list to csv
list0 = freq_dist_higher.most_common(100)
with open('topWords_allReviews.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    writer.writerows(list0)

print('======================================================================')

# Find words that are in both 5 and 1,2 star reviews
df1 = pd.DataFrame(list0, columns=['word', 'count'])
df1 = df1[['word']]

df2 = pd.DataFrame (list1, columns=['word', 'count'])
df2 = df2[['word']]  

df1 = df1.merge(df2, on='word')
df1.to_csv('wordsSimilar_34reviews.csv', index=False)      

del df1, df2, df1_clean, df2_clean

###############################################################################
# All reviews
df1 = pd.read_csv('topWords_allReviews.csv')
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

df10 = df9.dropna(how='any')
df11 = df10.iloc[:,[0]]
df11.to_csv('wordsSimilar_allreviewsMerge.csv', index=False)      

###############################################################################
# Sentiment
df = df[['stars_reviews', 'cleanReview1']]
df.rename(columns={'cleanReview1': 'cleanReview'}, inplace=True)
df = df.drop_duplicates()

df['polarity'] = df['cleanReview'].apply(lambda x: TextBlob(x).sentiment[0])

print(df['polarity'].describe().apply("{0:.3f}".format))

def getAnalysis(score):
  if score >= 0.4:
      return 'Positive'
  elif score > 0.2 and score < 0.4:
      return 'Slightly Positive'
  elif score <= 0.2 and score >= 0.0:
      return 'Slightly Negative'
  else:
      return 'Negative'

df['sentiment'] = df['polarity'].apply(getAnalysis)
df[['sentiment']].value_counts()

# Convert to binary
df['sentiment'].mask(df['sentiment'] == 'Negative', 0, inplace=True)
df['sentiment'].mask(df['sentiment'] == 'Positive', 1, inplace=True)

df['stars_reviews'].mask(df['stars_reviews'] == 1, 0, inplace=True)
df['stars_reviews'].mask(df['stars_reviews'] == 2, 0, inplace=True)
df['stars_reviews'].mask(df['stars_reviews'] == 5, 1, inplace=True)

# Sample equal for balanced sets
df1 = df[df.sentiment==1]
df1 = shuffle(df1)
df1 = df1.sample(n=414937)

df2 = df[df.sentiment==0]

sent = pd.concat([df1, df2])
del df1, df2

# Shuffle
sent = shuffle(sent)

# Write to parquet
sent = sent[['cleanReview', 'sentiment', 'stars_reviews']]
sent.to_parquet('sentimentNegPos.parquet')

###############################################################################
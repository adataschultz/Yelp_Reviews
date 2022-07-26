# Natural Language Understanding of Reviews


## Purpose
Which characteristics of businesses associate with a higher frequency of positive reviews?


## Questions
1)	Do the services provided by the business to the customer, hours of operation and time during the year of the business play any role in the amount of positive reviews?
2)	Does the activity of the individual who reviews businesses on the site demonstrate any patterns that could affect the increased number of reviews?
3)	Are the different components of reviews associated with any of the features provided and engineered? 


## Data
The data was retrieved from https://www.yelp.com/dataset. This includes business, review, user, tip and photo `json` files. The `photo.json` was not utilized in constructing a data warehouse. 


## Preprocessing
A data warehouse was constructed by joining the four sets utilized in the analysis. The `reviews` were used as the backbone of the warehouse in order to increase the number of reviews. The data was joined in the following order:
- `review`
- `business`
- `user`
- `tip`
- `checkin`

The `business` data was used to obtain the name of business in the `tip` set. This allowed for features to be engineered like the number of compliments. The `checkin` set allowed for features to be created based of the time information. Exploratory data analysis (EDA) was then completed to examine the various components of the warehouse including the reviews, users, business ratings, locations and time. Prior to the NLP processing step, the warehouse was filtered to the states with the seven highest counts of reviews and the food categories with over 30k counts. 

For the text processing step, all non-English words were removed prior to processing the reviews using `langid` for language detection and `dask` for parallel processing. Then, all non-words were removed. Then the reviews were processed for `stopwords` using `nltk` and lemmatized using `WordNetLemmatizer`. Further EDA was conducted after processing.


## Classification
- `Word2Vec` from `gensim` to create vocabulary -> Classification using `xgboost`, `catboost` and `lightgbm` on `GPU`.
- `tokenizer` from `keras` -> `Bidirectional LSTM` using `tensorflow`.
- Tokenization and Vectorization using `tf.keras.layers.TextVectorization` with `max_features = 50000` and `sequence_length = 300`. - > Classifier using layers consisting of `Embedding`, `GlobalAveragePooling1D`, with `Dropout(0.2)` after each layer with a final `Dense` layer. The baseline `Embedding` size is `embedding_dim = 32`.  The loss function utilized was `losses.BinaryCrossentropy`, `adam` as the optimizer and the default learning rate. The models were trained for 10 epochs using `BATCH_SIZE = 1`. Hyperparameter tuning using grid and search search using `wandb`.
- `BoW` and `TF-IDF` models were constructed in `pytorch` by sampling 20,000 of each target group, building a vocabulary of `MAX_LEN = 300` and `MAX_VOCAB = 10000`, tokenizing with `wordpunct_tokenize`, removing rare words and constructing `BoW vector` and `TF-IDF vector`. `BATCH_SIZE = 1` was used for both models. A `FeedfowardTextClassifier` with two hiddens layers was used for classification. The loss function utilized was `CrossEntropyLoss`, `adam` as the optimizer, `LEARNING_RATE = 6e-5` and the `scheduler = CosineAnnealingLR(optimizer, 1)`. Early stopping was used if the validation loss was greater than the validation loss from three previous epochs.
- A classifier utilizing`BertModel.from_pretrained('bert-base-uncased')` using `pytorch` was constructed using the `BertTokenizer.from_pretrained('bert-base-uncased')` with `max_length = 300`. The model architecture employed an initial `Dropout(p=0.4)` followed by a linear, another dropout, a `ReLU` and a final linear layer. The loss function was `CrossEntropyLoss` with `adam` optimizer and a `lr=5e-6`. `batch_size=1` and `batch_size=8` were tested in different models.
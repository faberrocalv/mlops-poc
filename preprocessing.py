# Import libraries
import nltk
nltk.download('punkt_tab')
nltk.download('popular')
import helpers
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from joblib import dump


data = pd.read_excel('./raw_data/sentimientos.xlsx')

# Tokenize text
data['tokens'] = data['comentario'].apply(lambda x: helpers.tokenize(x))

# Delete stopwords
data['sin_stopwords'] = data['tokens'].apply(lambda x: helpers.clean_stopwords(x))

# Root reduction
data['stemming'] = data['sin_stopwords'].apply(lambda x: helpers.stem_tokens(x))

# Represent as embeddings
tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=helpers.dummy_fun,
    preprocessor=helpers.dummy_fun,
    token_pattern=None)  

X_tfidf = tfidf.fit_transform(data['stemming'])
data_tfidf = pd.DataFrame(X_tfidf.todense(), columns=tfidf.get_feature_names_out())

# Add sentiment to tfidf matrix
data_tfidf['sentimiento'] = data['sentimiento']

# 70-30 Split
train, test = train_test_split(data_tfidf, test_size=0.3, stratify=data_tfidf['sentimiento'])

# Save processed data and emmbeddings
train.to_csv('./processed_data/train.csv', index=False)
test.to_csv('./processed_data/test.csv', index=False)
dump(tfidf, './model/tfidf.joblib')

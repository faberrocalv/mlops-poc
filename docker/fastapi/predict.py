# Import libraries
import nltk
nltk.download('punkt_tab')
nltk.download('popular')
import pandas as pd
from helpers import tokenize, clean_stopwords, stem_tokens, dummy_fun
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer


def predict_sentiment(comment):
    # Load model and embeddings
    tfidf = load('./app/jenkins_data/workspace/mlops-pipeline/tfidf.joblib')

    model = load('./app/jenkins_data/workspace/mlops-pipeline/class_model.joblib')
    
    # Process input comment
    data_fut = pd.DataFrame(data={'comentario': [comment]})

    data_fut['tokens'] = data_fut['comentario'].apply(lambda x: tokenize(x))

    data_fut['sin_stopwords'] = data_fut['tokens'].apply(lambda x: clean_stopwords(x))

    data_fut['stemming'] = data_fut['sin_stopwords'].apply(lambda x: stem_tokens(x))

    tfidf_fut = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        vocabulary=tfidf.vocabulary_
    )  # Se debe indicar el diccionario del aprendizae

    X_tfidf_fut = tfidf_fut.fit_transform(data_fut['stemming'])
    data_fut_tfidf = pd.DataFrame(X_tfidf_fut.todense(), columns=tfidf_fut.get_feature_names_out())

    # Predict
    Y_fut = model.predict(data_fut_tfidf)
    return Y_fut

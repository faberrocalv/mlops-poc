
# Import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

# Lower and delete special characters
def tokenize(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words


# Delete stopwords
def clean_stopwords(lista):
  clean_tokens = lista[:]
  sr = stopwords.words('spanish')
  for token in lista:
    if token in stopwords.words('spanish'):
      clean_tokens.remove(token)
  return clean_tokens


# Root reduction
def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem


def dummy_fun(doc):
    return doc

import re
import pandas
import nltk
import string
from datetime import datetime
from sklearn.model_selection import train_test_split
import nltk
import spacy
import string
#from nltk.tokenize import word_tokenize

# Opening JSON file as dataframe
df = pandas.read_csv('que-faire-a-paris-.csv', delimiter = ';')

#cleaning description
#df = df[:500]
df = df[df['Date de fin'] > str(datetime.today().strftime('%Y-%m-%d'))].reset_index(drop = True)

df['Description']=df[df.columns[2:7]].apply(lambda x:' '.join(x.astype(str)),axis=1)
description = df['Description'].apply(lambda x: re.sub('<[^<]+?>', ' ', x))

#text analysis

nltk.download('stopwords')
nltk.download('punkt')

# Removing punctuation

def Remove_Punct(text):
  result = "".join([ch for ch in text if ch not in string.punctuation])
  return result

description = description.to_frame()
description['Description'] = description['Description'].apply(lambda x: Remove_Punct(x))
description['Description'] = description['Description'].apply(lambda x: x.lower())
#nlp = fr_core_news_md.load()
#python -m spacy download fr au préalable dans un cmd

#Removing stopwords from descriptions

nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

def Remove_Stopwords(text):
  result = "".join([ch for ch in text if ch not in stopwords])
  return result

#tokenizing descriptions
description['tokenized'] = description['Description'].apply(lambda x: nlp(x))

# Lemmatize descriptions
def applyLemming(token):
  stemmedList=[]
  for word in token:
    if str(word).lower() not in stopwords:
      stemmedList.append(word.lemma_)
  return stemmedList

description["Lemmed"] = description['tokenized'].apply(lambda x: applyLemming(x))

pretrait=[]
for phrase in description['Description']:
  newphrase=[]
  for token in nlp(phrase):
    if not(token.is_stop):
      lem = token.lemma_
      newphrase.append(lem)
  pretrait.append(newphrase)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens ans return best mathes.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index


def get_recommendations_tfidf(sentence, tfidf_mat):

  """
  Return the database sentences in order of highest cosine similarity relatively to each 
  token of the target sentence. 
  """
  # Embed the query sentence
  tokens_query = [str(tok) for tok in nlp(sentence)]
  embed_query = vectorizer.transform(tokens_query)
  # Create list with similarity between query and dataset
  mat = cosine_similarity(embed_query, tfidf_mat)
  # Best cosine distance for each token independantly
  best_index = extract_best_indices(mat, topk=3)
  response = "Voici nos recommandations : " + "\n\n1."+ str(df.loc[best_index[0], 'Titre']) + "\nCliquez sur lien : " + str(df.loc[best_index[0], 'URL']) + "\n\n2."+ str(df.loc[best_index[1], 'Titre']) + "\nCliquez sur lien : " + str(df.loc[best_index[1], 'URL']) +"\n\n3."+ str(df.loc[best_index[2], 'Titre']) + "\nCliquez sur lien : " + str(df.loc[best_index[2], 'URL'])
  print(response)

import pickle
filename = 'td_idf_mat_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Fit TFIDF
vectorizer = TfidfVectorizer() 
tfidf_mat = vectorizer.fit_transform(description['Description']) 

# Return best threee matches between query and dataset
test_sentence = 'concert rock' 
get_recommendations_tfidf(test_sentence, tfidf_mat)


"""

# save the model to disk

pickle.dump(tfidf_mat, open(filename,"wb"))


"""


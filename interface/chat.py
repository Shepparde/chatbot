import re
import pandas
import nltk
import string
from datetime import datetime
from sklearn.model_selection import train_test_split
import nltk
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load base DataFrame
df = pandas.read_csv("que-faire-a-paris-.csv",sep=";")
df = df[df['Date de fin'] > str(datetime.today().strftime('%Y-%m-%d'))].reset_index(drop = True)

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
    print('Meilleur indice extrait')
    return best_index

def get_recommendations_tfidf(sentence, tfidf_mat):

  """
  Return the database sentences in order of highest cosine similarity relatively to each 
  token of the target sentence. 
  """
  # Embed the query sentence 
  nlp = spacy.load("fr_core_news_sm")
  tokens_query = [str(tok) for tok in nlp(sentence)]
  embed_query = vectorizer.transform(tokens_query)
  # Create list with similarity between query and dataset
  mat = cosine_similarity(embed_query, tfidf_mat)
  # Best cosine distance for each token independantly
  if np.mean(mat) > 0.0005:
    best_index = extract_best_indices(mat, topk=3)
    response = "Voici nos recommandations : " + "<br><br>1."+ str(df.loc[best_index[0], 'Titre']) + "<br>" + "<a target=”_blank” href="+str(df.loc[best_index[0], 'URL'])+">Cliquez ici</a>" + "<br><br>2."+ str(df.loc[best_index[1], 'Titre']) + "<br>" + "<a target=”_blank” href="+str(df.loc[best_index[1], 'URL'])+">Cliquez ici</a>" + "<br><br>3."+ str(df.loc[best_index[2], 'Titre']) + "<br>" + "<a target=”_blank” href="+str(df.loc[best_index[2], 'URL'])+">Cliquez ici</a>"
  else:
    response = "Nous ne trouvons pas de résultats à votre demande. Veuillez détailler votre demande."
  return response
  

#import pickle

#filename = 'tfidf_model.sav'

# load the model from disk
"""loaded_model = pickle.load(open(filename, 'rb'))"""

#Load Preprocessed DataFrame
description = pandas.read_csv("preprocessed_data.csv",sep=";")
print("Dataframe chargée")
# Fit TFIDF
vectorizer = TfidfVectorizer() #loaded_model
tfidf_mat = vectorizer.fit_transform(description['Lemmed_1']) 
print('Vectorisation effectuée')


# save the model to disk

#pickle.dump(vectorizer.vocabulary_, open(filename,"wb"))



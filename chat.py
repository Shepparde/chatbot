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

#training_sentences, test_sentences = train_test_split(df['Description'], test_size=0.8)
"""""
from gensim.models import Phrases
docs=pretrait
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=3, no_above=0.3)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)
"""
#top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
#avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
#print('Average topic coherence: %.4f.' % avg_topic_coherence)

# Topics of each data of the dataset
#df['topics'] = 0
#for c in corpus:
#  topics = model.top_topics(c)
#  df[df['Description'] ==c]['topics'] = topics

#print(df.head())

#from pprint import pprint
#pprint(top_topics)


# VERSION CHARLOTTE

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

# Fit TFIDF
vectorizer = TfidfVectorizer() 
tfidf_mat = vectorizer.fit_transform(description['Description']) 

# Return best threee matches between query and dataset
test_sentence = 'concert rock' 
get_recommendations_tfidf(test_sentence, tfidf_mat)


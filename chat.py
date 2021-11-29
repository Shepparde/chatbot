import re
import pandas
import nltk
import string
from datetime import datetime
from sklearn.model_selection import train_test_split
import spacy

# Opening JSON file as dataframe
df = pandas.read_csv('que-faire-a-paris-.csv', delimiter = ';')

#cleaning description
df = df[df['Date de fin'] > str(datetime.today().strftime('%Y-%m-%d'))]


description = df['Description'].apply(lambda x: re.sub('<[^<]+?>', '\t', x))

#text analysis

nltk.download('stopwords')
nltk.download('punkt')

def Remove_Punct(text):
  result = "".join([ch for ch in text if ch not in string.punctuation])
  return result

pct = list(string.punctuation)
description=description.to_frame()

description['Description'] = description['Description'].apply(lambda x: re.sub('\'\’', '\t', x))
description['Description'] = description['Description'].apply(lambda x: Remove_Punct(x))

#python -m spacy download fr au préalable dans un cmd

nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

pretrait=[]
for phrase in description['Description']:
  newphrase=[]
  for token in nlp(phrase):
    if not(token.is_stop):
      lem = token.lemma_
      newphrase.append(lem)
  pretrait.append(newphrase)

#training_sentences, test_sentences = train_test_split(df['Description'], test_size=0.8)

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

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)
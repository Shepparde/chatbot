import re
import pandas
import nltk
import string
from datetime import datetime
from sklearn.model_selection import train_test_split
import nltk
import spacy
import string


# Removing punctuation

def Remove_Punct(text):
  result = "".join([ch for ch in text if ch not in string.punctuation])
  return result



#Removing stopwords from descriptions
nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

def Remove_Stopwords(text):
  result = "".join([ch for ch in text if ch not in stopwords])
  return result


# Lemmatize descriptions
def applyLemming(token):
  stemmedList=[]
  for word in token:
    if str(word).lower() not in stopwords:
      stemmedList.append(word.lemma_)
  return stemmedList


def applyAllTreatments():
    
        # Opening  file as dataframe
    df = pandas.read_csv('que-faire-a-paris-.csv', delimiter = ';')

    #cleaning description
    #df = df[:500]
    df = df[df['Date de fin'] > str(datetime.today().strftime('%Y-%m-%d'))].reset_index(drop = True)

    df['Description']=df[df.columns[2:7]].apply(lambda x:' '.join(x.astype(str)),axis=1)
    description = df['Description'].apply(lambda x: re.sub('<[^<]+?>', ' ', x))

    #text analysis

    nltk.download('stopwords')
    nltk.download('punkt')
    
    description = description.to_frame()
    description['Description'] = description['Description'].apply(lambda x: Remove_Punct(x))
    description['Description'] = description['Description'].apply(lambda x: x.lower())
    print('Ponctuation supprimée')

    description['Description'] = description['Description'].apply(lambda x: Remove_Stopwords(x))
    print('Stopwords supprimés')
    #tokenizing descriptions
    description['tokenized'] = description['Description'].apply(lambda x: nlp(x))

    description["Lemmed"] = description['tokenized'].apply(lambda x: applyLemming(x))
    description["Lemmed_1"] = description['Lemmed'].apply(lambda x: " ".join(x))
    print('Lemmatisé')
 
    description.to_csv("preprocessed_data.csv",sep=";")
    print('Dataframe Sauvegardée')

if __name__ == "__main__":
    applyAllTreatments()
else:
    print(__name__, "imported as module")
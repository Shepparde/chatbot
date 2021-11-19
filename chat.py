import re
import pandas
from datetime import datetime
from sklearn.model_selection import train_test_split

# Opening JSON file as dataframe
df = pandas.read_csv('que-faire-a-paris-.csv', delimiter = ';')

#cleaning description
df = df[df['Date de fin'] > str(datetime.today().strftime('%Y-%m-%d'))]
description = df['Description'].apply(lambda x: re.sub('<[^<]+?>', ' ', x))
training_sentences, test_sentences = train_test_split(df['Description'], test_size=0.8)

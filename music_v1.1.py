"""BOW"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pickle

df = pd.read_csv('dataset.csv')

df = df.dropna(axis=0)

lyrics = df['texto']
words = lyrics.str.lower().str.split()

set_words = set()
[set_words.update(word) for word in words]
vocabulary = dict(zip(set_words, range(len(set_words))))

def count_words(text, vocabulary):
    frequency = [0] * len(vocabulary)

    for word in text:
        if word in vocabulary:
            position = vocabulary[word]
            frequency[position] += 1
    return frequency


all_words_freq = [count_words(word, vocabulary) for word in words]

x = np.array(all_words_freq)
y = np.array(df['classificacao'])
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=0)

#Teste de Modelos
def predict(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=10)
    sucess_rate = np.mean(scores)
    return sucess_rate

all_results = {}

# Modelo GaussianNB
#model = GaussianNB()
#all_results['GaussianNB'] = predict('GaussianNB', model, x_train, y_train)

#model = MultinomialNB()
#all_results['MultinomialNB'] = predict('MultinomialNB', model, x_train, y_train)

model = AdaBoostClassifier()
all_results['AdaBoostClassifier'] = predict('AdaBoostClassifier',
                                            model, x_train, y_train)

for key, value in all_results.items():
    print(f"Modelo: {key} - Taxa de acerto: {value}")

## Treinar o AdaBoostClassifier para producao
model.fit(x_train, y_train)


# ## Export para producao
with open('vocabulary.pkl', 'wb') as file:
    pickle.dump(vocabulary, file)

with open('model_adaboost.pkl', 'wb') as file:
    pickle.dump(model, file)

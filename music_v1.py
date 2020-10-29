"""BOW"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('dataset.csv')

# Estudo do df
df.describe()
df.head(5)
df.isnull().sum()
df[df.classificacao.isnull()]

# Limpeza do df
df = df.dropna(axis=0)

# ## Preprocessamento - Bag of Words
lyrics = df['texto']
words = lyrics.str.lower().str.split()

# ## Criação de vocabulário
set_words = set()
[set_words.update(word) for word in words]

vocabulary = dict(zip(set_words, range(len(set_words))))

# Criar uma Funcao que:
# A. conta a presenca de cada palavra única
# B. Cria uma matriz de posições relativa às posições do vocabulario, e
#    computa a frequencia, de um texto passado.
#
# Ex: ola mundo, mundo azul -> a frequência de  mundo é 2.

def count_words(text, vocabulary):
    frequency = [0] * len(vocabulary)

    for word in text:
        if word in vocabulary:
            position = vocabulary[word]
            frequency[position] += 1
    return frequency


all_words_freq = [count_words(word, vocabulary) for word in words]
# ## Final do Preprocessamento

# ## Treinar o modelo
# Separar previsores (dados de entrada - x) e classe (y)
x = np.array(all_words_freq)
y = np.array(df['classificacao'])

# Dados de treinamento e teste
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=0)

# Treina o modelo GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Previsao
result = classifier.predict(x_test)
score = accuracy_score(result, y_test)

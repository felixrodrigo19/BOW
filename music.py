"""Producao"""
import numpy as np
import pickle


with open('resources/vocabulary.pkl', 'rb') as file:
    vocabulary = pickle.load(file)

with open('resources/model_adaboost.pkl', 'rb') as file:
    model = pickle.load(file)

def count_words(text, vocabulary):
    frequency = [0] * len(vocabulary)

    for word in text:
        if word in vocabulary:
            position = vocabulary[word]
            frequency[position] += 1
    return frequency


new_lyric = input("Insira a letra da música: ")
new_word = new_lyric.lower().split()
new_freq = count_words(new_word, vocabulary)

x = np.array(new_freq).reshape(1, -1)
if model.predict(x) == 1:
    result = "the Beatles"
else:
    result = "The Rolling Stones"
print(result)

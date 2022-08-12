import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import spacy

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

colnames = ["tweet_id", "entity", "label", "text"]
train = pd.read_csv("twitter_training.csv", header=0, names=colnames).dropna()
validation = pd.read_csv("twitter_validation.csv", header=0, names=colnames).dropna()

ohe = OneHotEncoder(sparse=False)
train_entities = np.array(train.entity)
validation_entities = np.array(validation.entity)
train_entities_oh = ohe.fit_transform(train_entities[:,None])
validation_entities_oh = ohe.transform(validation_entities[:,None])

nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "parser", "lemmatizer"])

spacied_train = list(nlp.pipe(train.text))
spacied_validation = list(nlp.pipe(validation.text))

#train_vectors = np.array([x.vector for x in spacied_train])
#validation_vectors = np.array([x.vector for x in spacied_validation])

# ignore stop words
def get_average_vectors(spacy_output):
    vectors = []
    for doc in spacy_output:
        # average stuff
        tok_vecs = [tok.vector for tok in doc if not tok.is_stop]
        if len(tok_vecs) == 0:
            mean_vec = np.zeros(301)
            mean_vec[-1] = 1
        elif len(tok_vecs) == 1:
            mean_vec = tok_vecs[0]
            mean_vec = np.concatenate((mean_vec, np.zeros(1)))
        else:
            mean_vec = np.mean(tok_vecs, axis=0)
            mean_vec = np.concatenate((mean_vec, np.zeros(1)))
        vectors.append(mean_vec)
    return np.array(vectors)

train_vectors = get_average_vectors(spacied_train)
validation_vectors = get_average_vectors(spacied_validation)

train_input = np.concatenate((train_vectors, train_entities_oh), axis=1)
validation_input = np.concatenate((validation_vectors, validation_entities_oh), axis=1)

model = LogisticRegression(max_iter=10000)
model.fit(train_input, train.label)

train_acc = model.score(train_input, train.label)
valid_acc = model.score(validation_input, validation.label)
print(train_acc)
print(valid_acc)

import pdb; pdb.set_trace()

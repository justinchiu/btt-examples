# Example script for majority class baseline
import numpy as np
import pandas as pd

colnames = ["tweet_id", "entity", "label", "text"]
train = pd.read_csv("twitter_training.csv", header=0, names=colnames)
validation = pd.read_csv("twitter_validation.csv", header=0, names=colnames)

labels, counts = np.unique(train["label"], return_counts=True)
idx = counts.argmax()
label = labels[idx]
print("Labels")
print(labels)
print("Counts")
print(counts)
print(f"Majority class: {label}")
print()

print("Majority class baseline")
train_accuracy = (train["label"] == label).sum() / train.shape[0]
validation_accuracy = (validation["label"] == label).sum() / validation.shape[0]
print(f"Train accuracy: {train_accuracy}")
print(f"Validation accuracy: {validation_accuracy}")
print()

print("Entity-dependent majority class baseline")
entities = train["entity"].unique()
ents_oh = np.array(train["entity"])[:,None] == entities
labelslist = list(labels)
labelmap = {x: 1+labelslist.index(x) for x in labels}
idmap = {1+labelslist.index(x): x for x in labels}
trainlabels = train["label"].map(lambda x: labelmap[x])
label_ent = np.array(trainlabels)[:,None] * ents_oh

# Map from query entity to its majority class
majority = {}
for i, e in enumerate(entities):
    labels, counts = np.unique(train["label"][ents_oh[:,i]], return_counts = True)
    idx = counts.argmax()
    label = labels[idx]
    majority[e] = label

train_accuracy = (
    train["label"] == train["entity"].map(lambda x: majority[x])
).sum() / train.shape[0]
validation_accuracy = (
    validation["label"] == validation["entity"].map(lambda x: majority[x])
).sum() / validation.shape[0]
print(f"Train accuracy: {train_accuracy}")
print(f"Validation accuracy: {validation_accuracy}")

# There is a decent boost in accuracy from the class-dependent baseline.
print(majority)
# However, the class-dependent majority baseline does not account for the text.
# The next step is to better condition on the text.


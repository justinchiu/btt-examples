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

train_accuracy = (train["label"] == label).sum() / train.shape[0]
validation_accuracy = (validation["label"] == label).sum() / validation.shape[0]

print(f"Train accuracy: {train_accuracy}")
print(f"Validation accuracy: {validation_accuracy}")

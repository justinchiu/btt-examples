# Example huggingface fine-tuning script
# Based on https://huggingface.co/docs/transformers/training#
 
# Requirements:
# Run
# pip install transformers, datasets
# to install the HuggingFace transformers and datasets libraries
 
import numpy as np

# Huggingface Datasets
from datasets import load_dataset, load_metric

# Our goal is to do entity-specific sentiment classification,
# following https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis.
# Download the data from there into the same directory as this script.

# HF API: one of the columns must be called "label", to be used during training.
colnames = ["tweet_id", "entity", "label", "text"]
data_files = dict(train="twitter_training.csv", validation="twitter_validation.csv")
dataset = load_dataset("csv", data_files=data_files, names=colnames)
# See https://huggingface.co/docs/datasets/loading for details.

print(dataset["train"][0])

# Huggingface Transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Loads a pretrained tokenizer that preprocesses (tokenize, vectorize) words
# for us. Make sure that the tokenizer used aligns with the model you end up using.
MODELNAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)

# This function will be applied to every example in the dataset.
def tokenize_function(examples):
    # Our input is the text and entity.
    # In order to combine these two, we want to concatenate them by adding the [SEP]
    # separater token between them.
    # All inputs are assumed to start with the special [CLS] token,
    # a relic we inherit from pretrained models such as BERT.
    batch_sentences = [
        f"[CLS] {x} [SEP] {y}"
        for x,y in zip(examples["text"], examples["entity"])
    ]
    return tokenizer(
        batch_sentences,
        padding="max_length",
        truncation=True,
    )

# We can apply the tokenization function to all examples in our dataset here.
tokenized_datasets = (dataset
    .map(tokenize_function, batched=True)
    # We have to encode the label column, which was originally just a string
    .class_encode_column("label")
)
# Use a subsampled small training dataset to speed things up
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# remove the call to select to run on the full training data, which may take a long time
eval_dataset = tokenized_datasets["validation"]

num_labels = np.unique(dataset["train"]["label"]).shape[0]
model = AutoModelForSequenceClassification.from_pretrained(MODELNAME, num_labels=num_labels)

# Evaluate the model with accuracy of the top-1 model prediction,
# as stated on the Kaggle page.
metric = load_metric("accuracy")

# The evaluation function to be passed to the trainer.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# HuggingFace trainer + arguments
# See https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/trainer#transformers.Trainer
# for details.
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = small_train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics=compute_metrics,
)
# Running this will fine-tune the model
trainer.train()

# Be sure to check out the training graphs on the wandb link!

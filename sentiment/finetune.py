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

# There are a variety of pretrained large language models available for use.
# We use a model that was 1) pretrained on a large amount of data (called RoBERTA)
# that was also 2) fine-tuned on a pre-existing sentiment task.
# Since this model was already fine-tuned on sentiment, we hope that it will
# adapt very quickly to our new sentiment task.
MODELNAME = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
# For more details, see https://huggingface.co/docs/transformers/preprocessing.

# This function will be applied to every example in the dataset.
def tokenize_function(examples):
    # Our input is the text and entity.
    # In order to combine these two, we want to concatenate them by adding the [SEP]
    # separater token between them.
    batch_sentences = [
        f"{x} [SEP] {y}"
        for x,y in zip(examples["text"], examples["entity"])
    ]
    return tokenizer(
        batch_sentences,
        padding = True,
        truncation = True,
    )

# We have to encode the label column, which was originally just a string
datasets = dataset.class_encode_column("label")

# Use a subsampled small training dataset to speed up training.
# The model is very large and slow to finetune.
# Please see other models for cheaper approaches.
BATCHSIZE = 16
small_train_dataset = (datasets["train"]
    .shuffle(seed=42)
    .select(range(BATCHSIZE*900))
    .map(
        tokenize_function,
        batched = True,
        batch_size = datasets["train"].shape[0],
    )
)
# remove the call to select to run on the full training data, which may take a long time
eval_dataset = datasets["validation"].map(
    tokenize_function,
    batched = True,
    batch_size = datasets["validation"].shape[0],
)

num_labels = np.unique(dataset["train"]["label"]).shape[0]
model = AutoModelForSequenceClassification.from_pretrained(
    MODELNAME,
    num_labels = num_labels,
    ignore_mismatched_sizes = True,
)

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
training_args = TrainingArguments(
    output_dir = "test_trainer",
    evaluation_strategy = "epoch",
    per_device_train_batch_size = BATCHSIZE,
    per_device_eval_batch_size = BATCHSIZE,
    learning_rate = 1e-5,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = small_train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
)
# Running this will fine-tune the model
trainer.train()

trainer.evaluate()
# Be sure to check out the training graphs on the wandb link!
# An example run: https://wandb.ai/justinchiu/huggingface/runs/2he8fz5b/overview

# Interestingly, this does worse than some simple baselines:
# https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc
# It's likely that the model needs more tuning (which we will not be doing due to the
# model being too slow).
 

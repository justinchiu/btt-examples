# Sentiment analysis example
In this repo we will run through sentiment analysis using a variety of approaches.
Our goal is to classify the sentiment of a tweet with respect to a particular entity.
We will use the data from the
[Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
competition on Kaggle, which also contains some data examples.

The extracted csv files are given in: `twitter_training.csv` and `twitter_validation.csv`.

We implement
* A majority class baseline: `python majority.py`.
    We also run an entity-dependent majority class baseline.
* A fine-tuned [RoBERTa](https://arxiv.org/abs/1907.11692) model: `python finetune.py`.
    This model requires a lot of compute to run, and took around 7 hours to train on a
    GPU with a batch size of 16.
    This model obtained 98% validation accuracy.
* (TBD) A bag-of-words model with different word representations: `python bagofwords.py`.
    We reproduce the results of this
    [Kaggle notebook](https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc),
    which reported 91% validation accuracy with a LR baseline and 93% with a neural bag of words model.

See the Results section below for a summary of validation accuracies.

## Dependencies
We highly recommend first installing [Anaconda](https://docs.anaconda.com/anaconda/install/)
and using a
[virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

* [Spacy](https://spacy.io/usage)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
* numpy

## Results
| Model                           | Validation Accuracy |
| ------------------------------- | ------------------- |
| Majority class                  |                 26% |
| Entity-dependent majority class |                 41% |
| Fine-tuned (sentiment) RoBERTa  |                 98% |
| LR on mean-pooled GloVe         |                 TBD | 

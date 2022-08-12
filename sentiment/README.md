# Sentiment analysis example
In this repo we will run through sentiment analysis using a variety of approaches.
Our goal is to classify the sentiment of a tweet with respect to a particular entity.
We will use the data from the
[Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
competition on Kaggle, which also contains some data examples.

The extracted csv files are given in: `twitter_training.csv` and `twitter_validation.csv`.

We implement
* A majority class baseline: `python majority.py`
* A (very slow) fine-tuned [RoBERTa](https://arxiv.org/abs/1907.11692) model: `python finetune.py`,
    which gets 98% accuracy once trained on the full training data.
* A bag-of-words model with different word representations: `python bagofwords.py`.
    We reproduce the results of this [Kaggle notebook](https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc).

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

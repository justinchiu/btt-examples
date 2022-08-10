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
    which we find to be out-performed by the models trained this
    [Kaggle notebook](https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc).
* A bag-of-words model with different word representations: `python bagofwords.py`.
    Here we reproduce the analysis from the Kaggle notebook in order to debug
    the fine-tuned RoBERTa model.

## Dependencies
We highly recommend first installing [Anaconda](https://docs.anaconda.com/anaconda/install/)
and using a
[virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

* [Spacy](https://spacy.io/usage)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
* numpy


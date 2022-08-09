# Sentiment analysis example
In this repo we will run through sentiment analysis using a variety of approaches.
Our goal is to classify the sentiment of a tweet with respect to a particular entity.
We will use the data from the
[Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
competition on Kaggle, which also contains some data examples.

We provide the extracted csv files here: `twitter_training.csv` and `twitter_validation.csv`.

We will implement
* A majority class baseline: `python baseline.py`
* A bag-of-words baseline with different word representations (TODO)
* A fine-tuned BERT model: `python finetune.py`
 
and run through the data preprocessing necessary for each of these.

## Dependencies
We highly recommend first installing [Anaconda](https://docs.anaconda.com/anaconda/install/)
and using a
[virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

* [Spacy](https://spacy.io/usage)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
* numpy


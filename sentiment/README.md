# Sentiment analysis example
We give a sentiment-analysis example based on tweets.
Our goal is to classify the sentiment of a tweet with respect to a particular entity.
We will use the data from the
[Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
competition on Kaggle, which also contains some examples.

We provide the extracted csv files here: `twitter_training.csv` and `twitter_validation.csv`.

We will implement
* A majority class baseline (TODO)
* A bag-of-words baseline with different word representations (TODO)
* A fine-tuned BERT model
 
and run through the data preprocessing necessary for each of these.

## Dependencies
* [Spacy](https://spacy.io/usage)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
* numpy

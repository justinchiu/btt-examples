# Sentiment analysis example
In this example we will implement
* A majority class baseline
* A bag-of-words baseline with different word representations
* A RoBERTA baseline pulled from HuggingFace and fine-tuned on our data
and run through the data preprocessing necessary for each of these.

The data can be downloaded from the
[Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
competition on Kaggle.

We provide the extracted csv files here: `twitter_training.csv` and `twitter_validation.csv`.


## Dependencies
* [Spacy](https://spacy.io/usage)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
* numpy
* jax

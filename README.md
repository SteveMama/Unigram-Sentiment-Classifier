# Sentiment Classifier Project

This project implements a **sentiment classifier** that predicts whether a movie review is positive or negative, using linear classifiers. It includes data preprocessing, feature extraction, and evaluation on real-world data.

## Features:
- **Preprocessing**: Tokenization, stopword removal, and lemmatization using `nltk` or `spacy`.
- **Feature Extraction**: Unigram, bigram, and custom features.
- **Classification**: Linear models like Logistic Regression.

## Setup:
   ```bash
`python sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM`

`python sentiment_classifier.py --model LR --feats UNIGRAM`

`python sentiment_classifier.py --model PERCEPTRON --feats BETTER`

`python sentiment_classifier.py --model LR --feats BETTER`
   ```

## Requirements:
- Python 3.x
- `numpy`, `nltk`, `spacy`, `scikit-learn`

## Data:
Movie review dataset (positive/negative labels).

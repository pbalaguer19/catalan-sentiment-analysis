#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk, math, sys
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
sys.path.insert(0, '/home/pau/CatalanTwitterFeelingAnalysis/src/')
from tweetCleaner import TweetCleaner
from corpusReader import CorpusReader
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cleaner = TweetCleaner();

def main():
    sentences, polarity = getCorpus('./corpus/corpus.txt')
    trainAndTest(sentences, polarity)


def tokenize(text):
    """
    Tokenize function for the classifier model.
    """
    try:
        cleaned_text = cleaner.clean(text)
        stems = word_tokenize(cleaned_text)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

def getCorpus(filename):
    """
    Reads the corpus and shuffle it.
    Returns two lists: sentences and polarities.
    """
    sentences = []
    polarity = []
    reader = CorpusReader(filename)
    tweetsDict = reader.read_corpus()

    keys = list(tweetsDict.keys())
    random.shuffle(keys)
    for key in keys:
        sentences.append(key)
        polarity.append(tweetsDict[key])

    return sentences, polarity

def trainAndTest(sentences, polarity):
    """
    Trains the classifier and test it.
    """

    vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = tokenize,
        lowercase = True,
        max_features = 85
    )

    cutoff = int(math.floor(len(sentences)*3/4))

    corpus = {'Text' : sentences, 'Sentiment' : polarity}
    dfCorpus = pd.DataFrame(corpus)

    corpus_data_features = vectorizer.fit_transform(dfCorpus.Text.tolist())

    corpus_data_features_nd = corpus_data_features.toarray()

    X_train, X_test, y_train, y_test  = train_test_split(
            corpus_data_features_nd,
            dfCorpus.Sentiment,
            train_size=0.75,
            random_state=1234)

    tree = DecisionTreeClassifier(criterion='entropy',
                                        max_depth=5,
                                        random_state=0)
    tree = tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    print classification_report(y_test, y_pred)

if __name__ == '__main__':
    main()

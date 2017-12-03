#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk, math, sys
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
sys.path.insert(0, '/home/pau/CatalanTwitterFeelingAnalysis/src/')
from tweetCleaner import TweetCleaner
from corpusReader import CorpusReader
import random

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
    cutoff = int(math.floor(len(sentences)*3/4))

    pipeline = Pipeline([
        ('vect', CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                min_df = 50,
                max_df = 1.9,
                ngram_range=(1, 1),
                max_features=1000
                )),
        ('cls', LinearSVC(C=.2, loss='squared_hinge',max_iter=1000,multi_class='ovr',
                 random_state=None,
                 penalty='l2',
                 tol=0.0001
                 )),
    ])

    train = {'sentences' : sentences[:cutoff], 'polarity' : polarity[:cutoff]}
    dfTrain = pd.DataFrame(train)
    test = {'sentences' : sentences[cutoff:], 'polarity' : polarity[cutoff:]}
    dfTest = pd.DataFrame(test)

    # Train
    pipeline.fit(dfTrain.sentences, dfTrain.polarity)

    # Test
    predicted = pipeline.predict(dfTest.sentences)
    print classification_report(dfTest.polarity, predicted)


if __name__ == '__main__':
    main()

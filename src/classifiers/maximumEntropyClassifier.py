#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, math, collections, itertools
import nltk, nltk.classify.util
from nltk.metrics import *
from nltk.classify import MaxentClassifier

import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, '/home/pau/CatalanTwitterFeelingAnalysis/src/')
from tweetCleaner import TweetCleaner

reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    trainFeatures, testFeatures = getTrainAndTestFeatures(make_full_dict, './corpus/corpus.txt')
    evaluate_features(trainFeatures, testFeatures)

def make_full_dict(words):
    return dict([(word, True) for word in words])

def getTrainAndTestFeatures(feature_select, trainingFile):
    """
    Reads the corpus and split it.
    Returns the training and testing part.
    """
    cleaner = TweetCleaner()

    #reading pre-labeled input and splitting into lines
    posSentences = []
    negSentences = []

    with open(trainingFile, 'r') as f:
        for line in f.readlines():
            if line.startswith('1'):
                posSentences.append(cleaner.clean(line[2:]))
            elif line.startswith('0'):
                negSentences.append(cleaner.clean(line[2:]))

    posFeatures = get_features(posSentences, 'pos', feature_select)
    negFeatures = get_features(negSentences, 'neg', feature_select)

    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    return trainFeatures, testFeatures


def evaluate_features(trainFeatures, testFeatures):
    """
    Train and evaluate the classifier model.
    """
    classifier = MaxentClassifier.train(trainFeatures)

    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = classifier.classify(features)
            testSets[predicted].add(i)

    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print '**************************************'
    print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
    print 'pos fmeasure:', f_measure(referenceSets['pos'], testSets['pos'])
    print '**************************************'
    print 'neg precision:',precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
    print 'neg fmeasure:', f_measure(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(50)

def get_features(sentences, pol, feature_select):
    #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    features = []
    for i in sentences:
        words = unicode(i).split()
        words = [feature_select(words), pol]
        features.append(words)
    return features

if __name__ == '__main__':
    main()

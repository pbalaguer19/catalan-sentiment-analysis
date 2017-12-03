#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import xml.etree.ElementTree as ET


reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    if sys.argc < 2:
        print sys.argv[0] + ' tweetsfile'
        sys.exit(0)

    tweets_file = sys.argv[1]
    tweets = get_tweets_list(tweets_file)

    # EMOJIS
    print 'Reading emojis...'
    emojis = read_emojis()
    print 'Cleaning emojis...'
    emojis = clean_emojis(emojis)
    print 'Emoji dict len: {}'.format(len(emojis))

    # WORDS
    print 'Reading words...'
    words = read_words()
    print 'Words dict len: {}'.format(len(words))

    corpus = create_corpus(emojis, words, tweets)
    write_corpus(corpus)

def get_tweets_list(tfile):
    """
    Read the tweets of geo_**_twitterStatus_ca.tsv
    So it takes the second column
    """
    tweets = []
    with open(tfile, 'r') as f:
        for line in f.readlines():
            tweets.append(unicode(line.split('\t')[1]))
    return tweets

def read_emojis():
    """
    Reads the emojis from Emoji Sentiment Ranking paper.
    Calculates the score for each emoji and return it.
    """
    emojis = {}
    firstline = True
    with open('./data/Emoji Sentiment Ranking 1.0/Emoji_Sentiment_Data_v1.0.csv') as f:
        for line in f.readlines():
            if firstline:
                firstline = False
            else:
                parts = line.split(',')

                # The emojis with Total ocurrences < 5 are excluded
                if int(parts[2]) > 4:
                    # Negative score = Negative ocurrences / Total Ocurrences
                    neg = float(parts[4]) / float(parts[2])

                    # Positive score = Positive ocurrences / Total Ocurrences
                    pos = float(parts[6]) / float(parts[2])

                    # Final score = Positive score - Negative score
                    score = pos - neg

                    emojis[unicode(parts[0], "utf-8")] = score
    return emojis

def clean_emojis(emojis):
    """
    Remove emojis with |score| < 0.25
    """
    new_emojis = {}
    for emoji, score in emojis.iteritems():
        if score < -0.25:
            new_emojis[emoji] = score
        elif score > 0.25:
            new_emojis[emoji] = score
    return new_emojis

def read_words():
    """
    Reads the classified words from ML-SentiCon paper.
    Remove words with |score| < 0.50
    """
    root = ET.parse('./data/senticon.ca.xml').getroot()
    words = {}
    for word in root.iter('lemma'):
        if not isinstance(word.text, unicode):
            uni = unicode(word.text.replace(' ', '').replace('_', ' '), "utf-8")
        else:
            uni = word.text.replace(' ', '').replace('_', ' ')
        score = float(word.attrib['pol'])
        if score > 0.5:
            words[uni] = score
        elif score < -0.5:
            words[uni] = score
    return words


def create_corpus(emojis, words, tweets):
    """
    Return only the tweets with |score| > 0.20
    """
    corpus = {}
    for tweet in tweets:
        polarity = 0
        for w in tweet.split(' '):
            for emoji, score in emojis.iteritems():
                if emoji in w:
                    polarity = polarity + score

            for word, score in words.iteritems():
                if word in w:
                    polarity = polarity + score

        if polarity < -0.2:
            corpus[tweet] = 0
        elif polarity > 0.2:
            corpus[tweet] = 1

    return corpus

def write_corpus(corpus):
    """
    Writes the final corpus.
    """
    f = open('training_corpus_tweets_ca.txt', 'wb+')
    for tweet, polarity in corpus.iteritems():
        f.write(str(polarity) + '\t' + tweet + '\n')

if __name__ == '__main__':
    main()

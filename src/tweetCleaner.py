#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
# import Stemmer
import unicodedata
import string
import pyfreeling.freeling as freeling
import xml.etree.ElementTree as ET


reload(sys)
sys.setdefaultencoding('utf-8')

class TweetCleaner:
    """
    TweetCleaner class.
    Preprocessing toolkit:
        - Remove twitter elements
        - Check synonyms
        - Remove stopwords
        - Remove punctuation
        - Morphologycal analysis
    """
    def __init__(self):
        self._init_freeling()
        self._read_stopwords()
        self._read_senticon()
        self._read_dict()


    def _init_freeling(self):
        """
        Initialize all freeling variables
        """
        self.FREELINGDIR = "/usr/local"
        self.DATA = self.FREELINGDIR+"/share/freeling/"
        self.LANG="ca"
        freeling.util_init_locale("default")

        self._freeling_create_language_analyzer()
        self._freeling_create_analyzer()
        self._freeling_create_tagger_senseanotator_parser()


    def _freeling_create_language_analyzer(self):
        """
        Create language analyzer
        """
        self.la=freeling.lang_ident(self.DATA+"common/lang_ident/ident.dat")
        self.op= freeling.maco_options("ca")
        self.op.set_data_files( "",
                           self.DATA + "common/punct.dat",
                           self.DATA + self.LANG + "/dicc.src",
                           self.DATA + self.LANG + "/afixos.dat",
                           "",
                           self.DATA + self.LANG + "/locucions.dat",
                           self.DATA + self.LANG + "/np.dat",
                           self.DATA + self.LANG + "/quantities.dat",
                           self.DATA + self.LANG + "/probabilitats.dat")

    def _freeling_create_analyzer(self):
        """
        Create analyzers
        """
        self.tk=freeling.tokenizer(self.DATA+self.LANG+"/tokenizer.dat")
        self.sp=freeling.splitter(self.DATA+self.LANG+"/splitter.dat")
        self.sid=self.sp.open_session()
        self.mf=freeling.maco(self.op)

    def _freeling_create_tagger_senseanotator_parser(self):
        """
        Activate mmorpho odules to be used in next call.
        And then, create tagger, sense anotator, and parsers
        """
        self.mf.set_active_options(False, True, True, True,  # select which among created
                              True, True, False, True,  # submodules are to be used.
                              True, True, True, True )  # default: all created submodules are used

        self.tg=freeling.hmm_tagger(self.DATA+self.LANG+"/tagger.dat",True,2)
        self.sen=freeling.senses(self.DATA+self.LANG+"/senses.dat")
        self.parser= freeling.chart_parser(self.DATA+self.LANG+"/chunker/grammar-chunk.dat")
        self.dep=freeling.dep_txala(self.DATA+self.LANG+"/dep_txala/dependences.dat", self.parser.get_start_symbol())

    def _read_stopwords(self):
        """
        Reads the file with all Catalan Stopwords.
        """
        self.stopwords = []
        with open('./data/cat_stopwords.txt', 'r') as f:
            for line in f.readlines():
                word = line.replace('\r\n', '')
                self.stopwords.append(word.replace('\n', ''))

    def _read_senticon(self):
        """
        Reads the senticon file with polarities
        """
        root = ET.parse('./data/senticon.ca.xml').getroot()
        self.senticon_words = []
        for word in root.iter('lemma'):
            uni = word.text.replace(' ', '').replace('_', ' ')
            self.senticon_words.append(uni)

    def _read_dict(self):
        """
        Read the synonyms dict.
        """
        self.used_words = []
        self.cat_synonym = {}
        self.values = []
        self.key = ''
        with open('./data/cat_dict_sinonims.dat', 'r') as f:
            for line in f.readlines():
                if line.startswith('-|'):
                    self._get_dict_values(line)
                else:
                    self._get_dict_key(line)

    def _get_dict_values(self, line):
        """
        Gets the dict values (or synonyms)
        """
        for word in line.replace('-|', '').split('|'):
            word = re.sub(r'\[[^)]*\]','', word)
            word = re.sub(r'\([^)]*\)','', word)
            word = word.replace('\n', '')

            if ',' in word:
                for w in word.split(', '):
                    self.values.append(w)
            else:
                self.values.append(word)

    def _get_dict_key(self, line):
        """
        Gets the dict keys
        """
        if self.key != '' and (self.key in self.senticon_words \
            or self.key in self.stopwords) and self.key not in self.used_words:
            self._key_not_in_used_words()

        elif self.key in self.used_words:
            self._key_in_used_words()

        line = re.sub(r'\[[^)]*\]','', line)
        line = re.sub(r'\([^)]*\)','', line)
        self.key = line[:-3]
        if '|' in self.key:
            self.key = self.key[:-1]
        self.values = []

    def _key_not_in_used_words(self):
        """
        Called when a new key is added to the dict
        """
        if ',' in self.key:
            keys = self.key.split(',')
            for k in keys:
                self.cat_synonym[k] = self.values
                self.used_words.append(k)
        else:
            self.cat_synonym[self.key] = self.values
            self.used_words.append(self.key)

        for val in self.values:
            self.used_words.append(val)

    def _key_in_used_words(self):
        """
        Called when you need to add more values in specific key
        """
        if self.key in self.cat_synonym:
            for val in self.values:
                if val not in self.cat_synonym[self.key]:
                    self.cat_synonym[self.key].append(val)
                    self.used_words.append(val)
        else:
            for k,v in self.cat_synonym.iteritems():
                if self.key in v:
                    for val in self.values:
                        if val not in v:
                            self.cat_synonym[k].append(val)
                            self.used_words.append(val)

    def clean(self, tweet):
        """
        Main function.
        Data cleaner.
        """
        tweet = tweet.lower()
        tweet = self._remove_twitter_things(tweet)
        tweet = self._synonyms_dict(tweet)
        tweet = self._remove_stopwords(tweet)
        tweet = self._remove_punctuation(tweet)
        tweet = self._freeling_tweet(tweet + '.')
        return tweet

    def close(self):
        """
        Close the class object. After callig it, you can't call clean() function again.
        """
        self._close_freeling()

    def _remove_twitter_things(self, tweet):
        """
        Remove URLS, Usernames, Hashtag's tag and 'RT' text
        """
        if 'rt @' in tweet:
            tweet = tweet[3:]
        return re.sub(r"(?:\@|https?\://)\S+", "", tweet).replace('#', '')

    def _synonyms_dict(self, tweet):
        """
        Replace all the words in synonym dict values for the key
        """
        for word in tweet.split(' '):
            for k, val in self.cat_synonym.iteritems():
                if k != word and word in val:
                    tweet.replace(word, k)
        return tweet

    def _remove_stopwords(self, tweet):
        """
        Remove all stopwords and catalan apostrophes:
            l', d', m', s', t', n'
            'n, 'm, 't, 's, 'l, 'ls
        """
        apostrophes = ["l\xe2\x80\x99", "d\xe2\x80\x99", "m\xe2\x80\x99", \
                        "s\xe2\x80\x99", "t\xe2\x80\x99", "n\xe2\x80\x99", \
                        "\xe2\x80\x99n", "\xe2\x80\x99m", "\xe2\x80\x99t", \
                        "\xe2\x80\x99s", "\xe2\x80\x99l", "\xe2\x80\x99ls", \
                        "l'", "d'", "m'", "s'", "t'", "n'", \
                        "'n", "'m", "'t", "'s", "'l", "'ls"]

        for apostrophe in apostrophes:
            tweet = tweet.replace(apostrophe, '')
        #tweet = ' '.join([c for c in tweet.split() if not c.startswith(tuple(apostrophe))])
        return ' '.join([c for c in tweet.split() if c not in self.stopwords])

    def _remove_punctuation(self, tweet):
        """
        Remove all punctuation
        """
        non_words = list(string.punctuation)
        non_words.extend(['¿', '¡', '‘', '’'])
        non_words.extend(map(str,range(10)))

        return ''.join([c for c in tweet if c not in non_words])

    def _freeling_tweet(self, tweet):
        """
        Search the root word for each word in tweet.
        """

        l = self.tk.tokenize(tweet)
        ls = self.sp.split(self.sid,l,False)

        ls = self.mf.analyze(ls)
        ls = self.tg.analyze(ls)
        ls = self.sen.analyze(ls)
        ls = self.parser.analyze(ls)
        ls = self.dep.analyze(ls)

        words = []
        ## Output results
        for s in ls :
           ws = s.get_words()
           for w in ws :
              # print(w.get_form()+" "+w.get_lemma()+" "+w.get_tag()+" "+w.get_senses_string())
              lema = w.get_lemma()
              if lema.startswith('['):
                  lema = w.get_form()
              words.append(lema)

        return " ".join(words).replace(' .', '')

    def _close_freeling(self):
        """
        clean up
        """
        self.sp.close_session(self.sid)

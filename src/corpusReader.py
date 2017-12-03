#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

class CorpusReader:
    """
    CorpusReader class.
    It reads the corpus and return it.
    """

    def __init__(self, filename):
        self.filename = filename

    def read_corpus(self):
        """
        Reads the filename corpus classified with positive (1)
        or negative (0) tweets.
        """
        tweetsDict = {}
        with open(self.filename, 'r') as f:
            for line in f.readlines():
                if line.startswith('1'):
                    tweetsDict[line[2:]] = 1
                elif line.startswith('0'):
                    tweetsDict[line[2:]] = 0

        return tweetsDict

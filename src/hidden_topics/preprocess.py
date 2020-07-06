#!/usr/bin/python

# encoding=utf8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


NUMS = "0123456789"


def readFuncWords():
    stop_words = set(stopwords.words("portuguese"))
    stop_words.update(['que', 'até', 'esse', 'de', 'do','essa', 'pro', 'pra', 'oi', 'lá'])
    return stop_words


def tokenizeText(text_str):
    """
    return a list of tokens
    """
    func_tokens = readFuncWords()
    #text_str = text_str.decode("utf-8")
    text_str = text_str.lower().strip()
    tokens = nltk.word_tokenize(text_str)
    # remove single characters
    tokens = [w for w in tokens if len(w) > 1]
    # remove functional words
    tokens = [w for w in tokens if w not in func_tokens]
    # remove numbers
    refined_tokens = []
    for w in tokens:
        num_flag = False
        for num in NUMS:
            num_flag = (num in w)
            if (num_flag):
                break
        if (not num_flag):
            refined_tokens.append(w)
    #print "refined tokens:", refined_tokens
    return refined_tokens
    


    










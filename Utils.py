import csv
import re
import urllib
from urllib.parse import urlparse

import nltk
import numpy as np


def csvSmartReader(csvFileName: str, fields: []):
    csv_reader = csv.DictReader(open(csvFileName), fieldnames=fields)  # w+ file created if it doesnt exist
    # http://stackoverflow.com/questions/1466000/python-open-built-in-function-difference-between-modes-a-a-w-w-and-r
    try:
        has_header = csv.Sniffer().has_header(open(csvFileName).read(1024))
    except:
        has_header = False  # case of empty file

    if has_header:
        next(csv_reader, None)  # skip the headers
    return csv_reader


def csvSmartWriter(csvFileName: str, fields: []):
    csv_writer = csv.DictWriter(open(csvFileName, 'w+'), fieldnames=fields)
    csv_writer.writeheader()
    return csv_writer


def encodeURL(url):
    url = url.strip()
    print(url)
    url = urllib.parse.urlsplit(url)
    url = list(url)
    for index, url_fragment in enumerate(url):
        if 'http' not in url_fragment:
            url[index] = urllib.parse.quote(url_fragment)
    url = urllib.parse.urlunsplit(url)
    return url




def extractURLFeatures(url):
    '''
    Use url parsing library provided by Python plus a bit of regular expressions
    :param url:
    :return:
    '''
    url_info = urlparse(url)
    domain = " ".join(re.split(".", url_info.netloc))
    path = " ".join(re.split("/|-|[0-9]?", url_info.path))
    url_feature = domain + path
    return url_feature


def getSumVectors(text, embed_matrix):
    vec = np.zeros((300,), float)
    for token in nltk.word_tokenize(text):
        if token in embed_matrix.vocab:
            vec = vec + embed_matrix[token]
    return vec
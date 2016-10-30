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

def detectCountryCodeDifference(url_pair: tuple)->int:
    """

    :param url_pairs:
    :return: 1 if different country code is detected from the url pair
    """



    diff = 0
    assert len(url_pair) == 2
    url_a, url_b = url_pair


    url_a = url_a.replace("in/","")
    url_b = url_b.replace("in/", "")

    url_a = url_a.replace("linkedin","_")
    url_b = url_b.replace("linkedin","_")

    country_a=""
    country_b=""

    country_code_reader = csvSmartReader('CountryCode/data.csv', ["Name","Code"])
    for row in country_code_reader:
        code =  row["Code"].lower().strip()
        if re.search("([/]"+code+"[\W])|(\."+code+"[/|$])", url_a) != None:
            country_a = row["Name"].strip()
        if re.search("([/]"+code+"[\W])|(\."+code+"[/|$])", url_b) != None:
            country_b = row["Name"].strip()

    if country_a != country_b and country_a != "" and country_b != "":
        print(url_a,'->',country_a, url_b, '->',country_b)
        diff=1

    return diff


def write_result_file(results, filename):
    writer = csvSmartWriter(filename, ["Id", "Outcome"])
    id = 200
    for result in results:
        row_dict = {}
        row_dict["Outcome"] = result
        row_dict["Id"] = id
        writer.writerow(row_dict)
        id += 1


    return
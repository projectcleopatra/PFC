
#Global parameters
import csv
import ssl
import urllib
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from selenium import webdriver #The hood way to bypass LinkedIn restriction

import TrumpCard

path_chrome_driver = '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages'

train_pairs = "alta16_kbcoref_train_pairs.csv"
train_snippet = "alta16_kbcoref_train_search_results.csv"
train_labels = "alta16_kbcoref_train_labels.csv"

test_pairs = "alta16_kbcoref_test_pairs.csv"
test_labels = "" #Not availablee
test_snippet = "alta16_kbcoref_train_search_results.csv"


def csvSmartReader(csvFileName: str, fields: []):
    csv_reader = csv.DictReader(open(csvFileName), fieldnames = fields)
    next(csv_reader, None) #skip the headers
    return csv_reader


def openFilesAndMakeFeatures(url_file: str, label_file: str, snippet_file: str = None):
    '''

    :param url_file: filename of the file that contains URLs
    :param snippet_file: filename of the snippet file (might not be available)
    :return:
    '''
    dataset = None

    input_reader = csvSmartReader(url_file, ['id','AUrl','BUrl'] )
    label_reader = csvSmartReader(label_file, ['id','outcome'])


    for input, label in zip(input_reader, label_reader):
        assert input['id'] == label["id"]
        AUrl = input["AUrl"]
        BUrl = input["BUrl"]
        scrapeThatPage(AUrl)
        scrapeThatPage(BUrl)





    return dataset


def scrapeThatPage(url:str):
    url = url.strip()
    print(url)
    try:
        req = urllib.request.Request(url, data=None, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    })
        soup = BeautifulSoup(urlopen(req), 'html.parser')
        print("Easy!")
    except ValueError:
        url = "http://"+ url
        req = urllib.request.Request(url, data=None, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        })
        soup = BeautifulSoup(urlopen(req), 'html.parser')
        print("Http added")

    except urllib.error.HTTPError:
        print(url)
        driver = webdriver.Chrome(path_chrome_driver)
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        summary = soup.find('section', {"id": "summary"})
        print(summary.getText())
        print("Use the ghetto way: ", url)



    print("Success", soup.title)
    print("========================")
    return


def trainModelAndEval(traindataSet, test):
    results = []

    return results


def main():
    '''
    1. Open the training files and labels, make dataset
    2. Go through each sample in the dataset, train the model, leave a few for validation
    3. Use the model to predict

    average entire document ---> FastPage
    named entity of NLTK --> ditch or pick sentence / use Mahalanolis-distance

    :return:
    '''
    train_dataset = openFilesAndMakeFeatures(train_pairs, train_labels)
    #test_dataset = openFilesAndMakeFeatures(test_pairs)
    #results = trainModelAndEval(train_dataset, test_dataset)
    



    return


if __name__ == "__main__":
   main()
# Global parameters
import datetime
import itertools
import os
import pickle
import urllib
from pathlib import Path
from time import sleep
from urllib.request import urlopen

from bs4 import BeautifulSoup
from selenium import webdriver  # The hood way to bypass LinkedIn restriction

import LinkedIn
from Utils import csvSmartReader, csvSmartWriter, encodeURL

url_pair_schema = ['id', 'AUrl', 'BUrl']
label_file_schema = ['id', 'outcome']
data_file_schema = ['id', 'A_raw_page', 'A_structured_data', 'A_last_update_datetime', 'B_raw_page',
                    'B_structured_data', 'B_last_update_datetime']
snippet_file_schema = ['id', 'AUrl', 'ATitle', 'ASnippet', 'BUrl', 'BTitle', 'BSnippet']


train_snippet = "alta16_kbcoref_train_search_results.csv"
train_labels = "alta16_kbcoref_train_labels.csv"

train_pairs = "alta16_kbcoref_train_pairs.csv"
test_pairs = "alta16_kbcoref_test_pairs.csv"
test_labels = ""  # Not availablee
test_snippet = "alta16_kbcoref_test_search_results.csv"


class GetData:
    def __init__(self):

        self.linkedin_api = LinkedIn.quick_api('7530vp73eoybr6', 'oNQohDeiaRiYpuUF')
        self.twitter_api = None  # TODO

        return

    def openFilesAndUpdateData(self, url_file: str, label_file: str):
        '''

        :param url_file: filename of the file that contains URLs
        :param snippet_file: filename of the snippet file (might not be available)
        :return:
        '''

        input_reader = csvSmartReader(url_file, url_pair_schema)
        label_reader = csvSmartReader(label_file, label_file_schema)


        return




# End of class definition




def main():
    '''
    1. Open the training files and labels, make dataset
    2. Go through each sample in the dataset, train the model, leave a few for validation
    3. Use the model to predict

    average entire document ---> FastPage
    named entity of NLTK --> ditch or pick sentence / use Mahalanolis-distance

    :return:
    '''
    # Make sure that the files are not empty due to read/write operation errors
    assert os.stat(train_pairs).st_size != 0
    assert os.stat(train_labels).st_size != 0
    assert os.stat(test_pairs).st_size != 0
    g = GetData()
    g.openFilesAndUpdateData(train_pairs, train_labels)
    # test_dataset = openFilesAndMakeFeatures(test_pairs)
    # results = trainModelAndEval(train_dataset, test_dataset)




    return


if __name__ == "__main__":
    main()

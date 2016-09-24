# Global parameters
import csv
import datetime
import os
import urllib
from time import sleep
from urllib.request import urlopen, Request

import itertools

import pickle
from bs4 import BeautifulSoup
from pathlib import Path
from selenium import webdriver  # The hood way to bypass LinkedIn restriction

import LinkedIn

url_pair_schema = ['id', 'AUrl', 'BUrl']
label_file_schema = ['id', 'outcome']
data_file_schema = ['id', 'A_raw_page', 'A_structured_data', 'A_last_update_datetime', 'B_raw_page',
                    'B_structured_data', 'B_last_update_datetime']
train_snippet = "alta16_kbcoref_train_search_results.csv"
train_labels = "alta16_kbcoref_train_labels.csv"

train_pairs = "alta16_kbcoref_train_pairs.csv"
test_pairs = "alta16_kbcoref_test_pairs.csv"
test_labels = ""  # Not availablee
test_snippet = "alta16_kbcoref_train_search_results.csv"


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


class GetData:
    def __init__(self):

        self.linkedin_api = LinkedIn.quick_api('7530vp73eoybr6', 'oNQohDeiaRiYpuUF')
        self.twitter_api = None  # TODO

        return

    def openFilesAndUpdateData(self, url_file: str, label_file: str, snippet_file: str = None):
        '''

        :param url_file: filename of the file that contains URLs
        :param snippet_file: filename of the snippet file (might not be available)
        :return:
        '''

        input_reader = csvSmartReader(url_file, url_pair_schema)
        label_reader = csvSmartReader(label_file, label_file_schema)

        data_writer = csvSmartWriter("tempdatafile.csv", data_file_schema)
        data_file = url_file[:-4] + '_scraped_content'







        # Check if the file exists
        data_reader = None
        if Path(data_file).is_file():
            data_reader = csvSmartReader(data_file, data_file_schema)

        for input, label in itertools.zip_longest(input_reader, label_reader, fillvalue=None):

            # Load the stored data or create a new data structure if stored data are not available
            try:
                with open(data_file, 'rb') as pickle_file:
                    downloaded_data = pickle.load(pickle_file)  # id --> {[field:value]*}
            except (FileNotFoundError, EOFError):
                downloaded_data = dict()

            assert input['id'] == label["id"]

            id = input['id']
            print("Getting sample id:", input['id'])
            AUrl = input["AUrl"]
            BUrl = input["BUrl"]

            updated_data = dict()
            updated_data['id'] = input['id']
            # update the data file only when it was last updated more than a certain period of time




            if id not in downloaded_data \
                    or (datetime.datetime.now() - \
                    downloaded_data[id]['A_last_update_datetime']) \
                    > datetime.timedelta(days=4)\
                    or not downloaded_data[id]['A_raw_page'] \
                    or not downloaded_data[id]['A_structured_data'] \
                    or not updated_data['A_last_update_datetime'] \
                    :
                raw_pageA, structured_dataA, datetimeA = self.scrapeThatPage(AUrl)
                updated_data['A_raw_page'] = str(raw_pageA)
                updated_data['A_structured_data'] = structured_dataA
                updated_data['A_last_update_datetime'] = datetimeA

            else:
                updated_data['A_raw_page'] = downloaded_data[id]['A_raw_page']
                updated_data['A_structured_data'] = downloaded_data[id]['A_structured_data']
                updated_data['A_last_update_datetime'] = downloaded_data[id]['A_last_update_datetime']

            if id not in downloaded_data \
                    or (datetime.datetime.now() - \
                    downloaded_data[id]['B_last_update_datetime']) \
                    > datetime.timedelta(days=4) \
                    or not downloaded_data[id]['B_raw_page'] \
                    or not downloaded_data[id]['B_structured_data'] \
                    or not updated_data['B_last_update_datetime'] \
                    :
                raw_pageB, structured_dataB, datetimeB = self.scrapeThatPage(BUrl)
                updated_data['B_raw_page'] = str(raw_pageB)
                updated_data['B_structured_data'] = structured_dataB
                updated_data['B_last_update_datetime'] = datetimeB
            else:
                updated_data['B_raw_page'] = downloaded_data[id]['B_raw_page']
                updated_data['B_structured_data'] = downloaded_data[id]['B_structured_data']
                updated_data['B_last_update_datetime'] = downloaded_data[id]['B_last_update_datetime']

            downloaded_data[id] = updated_data #add the data into pickle
            data_writer.writerow(updated_data)
            with open(data_file, 'wb') as picke_file:
                pickle.dump(obj = downloaded_data, file = picke_file)


        # After writing the tempfile, delete the old data file, rename it to the deleted data file name
        os.replace("tempdatafile.csv", data_file+'.csv')



        return

    def scrapeThatPage(self, url: str) -> tuple:

        if '.linkedin.com/' in url:
            # do something

            while True:
                try:

                    structured_profile = LinkedIn.useLinkedInAPI(url, self.linkedin_api)
                except:
                    sleep(5)
                    continue
                break

            print("API returning:", structured_profile)
            scraped_profile = LinkedIn.scrapeLinkedInPage(url)
            print("Scraped LinkedIn result: ", scraped_profile.title)
            print("========================")
            return scraped_profile, structured_profile, datetime.datetime.now()
        elif '.twitter.com' in url:
            print()

        try:
            req = urllib.request.Request(url, data=None, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            })
            soup = BeautifulSoup(urlopen(req), 'html.parser')
            print("Easy!")
        except ValueError:

            url = "http://" + url  # https: should not be passed into quote

            req = urllib.request.Request(url, data=None, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            })
            soup = BeautifulSoup(urlopen(req), 'html.parser')
            print("Http added")


        except urllib.error.HTTPError:

            url = encodeURL(url)

            print(url)
            driver = webdriver.Chrome(os.getcwd() + '/chromedriver')
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html)
            driver.quit()
            print("Use the ghetto way: ", url)

        print("Success", soup.title)
        print("========================")
        return soup, "", datetime.datetime.now()


# End of class definition

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

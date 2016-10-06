#everything about the inverse document frequency
import os
import pickle
import re

import time
from random import randint

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class IDF:

    def __init__(self, token_counter:[]):
        self.token_counter = token_counter #regular frequency counter,

        self.idf_file_name = 'idf_counter.pickle'
        try:
            with open(self.idf_file_name, 'rb') as f:
                self.idf_counter = pickle.load(f)
        except:
            self.idf_counter = dict()


    def set_idf(self):
        ghetto_chrome = webdriver.Chrome(os.getcwd() + '/chromedriver')
        total_num = 30000000000
        for token, freq in self.token_counter:
            if token in self.idf_counter:
                continue
            else:

                ghetto_chrome.get("http://www.google.com")
                ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
                # This is so G! Credit: http://stackoverflow.com/questions/22739514/how-to-get-html-with-javascript-rendered-sourcecode-by-using-selenium

                search_box = ghetto_chrome.find_element_by_name('q')
                search_box.clear()
                search_box.send_keys(token, Keys.RETURN)
                time.sleep(randint(1,2))
                ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
                time.sleep(randint(10, 30))  # So Google doesn't block

                time.sleep(5)
                result_count = ghetto_chrome.find_element_by_xpath("//div[@id='resultStats']").text
                temp = re.sub(r"[\D]+\(+(.*)", '', result_count)
                idf = total_num/float(re.sub(r"[\D]", '', temp))
                print(idf)
                self.idf_counter[token] = idf

        ghetto_chrome.close()
        self.save_idf()

        return


    def save_idf(self):
        with open(self.idf_file_name, 'w+b') as f:
            self.idf_counter = pickle.dump(self.idf_counter, f)
        return


def main():
    dummy_token_counter = [('and', 121900), ('to', 119233), ('the', 71410), ('of', 68213), ('in', 61312), ('at', 47171), ('no', 45910)]
    idf = IDF(dummy_token_counter)
    idf.set_idf()

    return

if __name__ == "__main__":
    main()

import time
from selenium import webdriver
import os

class TwitterGrab:

    def __init__(self, webdriver):
        self.webdriver = webdriver


    def getBio(self, url)->str:
        self.webdriver.get(url)
        time.sleep(5)
        about = self.webdriver.find_element_by_xpath("//*[@class='ProfileHeaderCard']")
        desp = ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])
        desp = desp.replace("Verified account", "")
        desp = desp.split("Joined", 1)[0] #delete all the information about Joined on certain date

        print (desp)
        return desp


if __name__ == '__main__':
    tg = TwitterGrab(webdriver.Chrome('../chromedriver'))
    tg.getBio("https://twitter.com/RepWolfeMoore")

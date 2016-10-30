import csv
import os
import pickle

import itertools
import re
import numpy as np
import time


#Instantiate the developer authentication class

from selenium import webdriver
#http://stackoverflow.com/questions/24880288/how-could-i-use-python-request-to-grab-a-linkedin-page
from Utils import extractURLFeatures

linkedin_username = 'passion.gratefulness.focus@gmail.com'
linkedin_password = '8YE-CXp-q8X-qCG'
linkedin_login_page = 'https://www.linkedin.com/uas/login'
num_classes = 2



class LinkedInGrab:

    def __init__(self, webdriver):

        self.webdriver = webdriver



    def getCookies(self):
        return self.webdriver.get_cookies()


    def grabLinkedInDescription(self, url):
        print('Grabbing description for:', url)
        words =''
        time.sleep(3)
        self.webdriver.implicitly_wait(8)
        self.webdriver.refresh() #that causes error sometimes
        self.webdriver.get(url)
        self.webdriver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")



        #split "dsdss-tilman/333" into ['dsdss', 'tilman', '', '', '', '']

        full_name = None

        while full_name is None:
            try:
                # connect
                full_name = self.grabText(self.grabElement("//span[@class='full-name']"))
            except:
                time.sleep(12*60*60)
                pass


        if len(full_name)==0: #Name is such an important identifier that it shouldn't be empty
            return 'url error'
        location = self.grabText(self.grabElement("//*[@class='locality']"))
        industry = self.grabText(self.grabElement("//dd[@class='industry']//a"))
        headline = self.grabText(self.grabElement("//p[@class='title']"))
        summary =  self.grabText(self.grabElement("//div[@id='background-summary']"))

        education= self.grabText(self.grabElement("//tr[@id='overview-summary-education']"))
        current = self.grabText(self.grabElement("//tr[@id='overview-summary-current']"))
        """
        previous = self.grabText(self.grabElement("//tr[@id='overview-summary-past']"))


        experience = self.grabText(self.grabElement("//div[@id='background-experience']"))
        skills = re.sub(r"[0-9]", ' ', self.grabText(self.grabElement("//div[@id='background-skills']")))
        #list
        languages = [lan.text for lan in self.grabElements("//div[@id='languages-view']//ol//li//h4//span")]
        followings = [elm.text for elm in self.grabElements("//div[@id ='following-container']//p[@class='following-name']")]
        """
        words = " ".join([ full_name, location, industry, headline, summary,
                          education, current])#, previous, experience, skills]+followings+languages)

        print(words)
        print('='*20)
        return words.replace("\n|\t", " ")




    def grabElement(self, xpath):
        try:
            return self.webdriver.find_element_by_xpath(xpath)
        except:
            print(xpath, 'not available')
            return ''

    def grabElements(self, xpath):
        try:
            return self.webdriver.find_elements_by_xpath(xpath)
        except:
            print(xpath, 'not available')
            return []

    def grabMoreLinkedInTrainingExamples(self, url):
        '''

        :param url:
        :return:
        '''
        #load a page
        urls = []
        urls.append(url)
        training_set=set()
        training_data = []
        time.sleep(3)
        self.webdriver.implicitly_wait(8)
        self.webdriver.refresh()
        self.webdriver.get(url)
        self.webdriver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        time.sleep(3)
        self.webdriver.implicitly_wait(8)
        #get the "people you might know" field
        other_profiles = self.grabElements("//ol[@class='discovery-results']//li//dl/dt/a")


        #get the "People also viewed...." field
        other_profiles.extend(self.grabElement("//ul[@class='browse-map-lists']/li/a"))

        for element in other_profiles:
            urls.append(element.get_attribute('href'))
        #make training example set
        for profile_link in urls:
            desc = self.grabLinkedInDescription(profile_link)
            if desc == 'url error':
                continue #don't collect the profile when it's bad
            training_set.add(desc)

        #from the traning_set, randomly select two links and make training examples
        if len(training_set)>1:
            print(len(training_set))
            unlabelled_training_set = list(itertools.combinations(training_set, 2))
            for description1, description2 in unlabelled_training_set:
                training_data.append( (description1, description2, "0" ) ) #0 when the pair refer to a different entities
            return training_data
        else:
            return []

    def harvestLinkedInLinksFromFile(self, url_file_name)->[]:

        linkedin_urls = []
        with open(url_file_name) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                for cell in row:
                    if 'linkedin.com/' in cell: # a crude way to tell whether it's a linkedin url or not
                        linkedin_urls.append(cell)
                        print(cell)
        return linkedin_urls

    def harvestDeterministicLinkedInExamples(self, linkedin_urls: list):
        '''
        For every linkedin url, iteratively grab a bunch of related urls and create training samples with them

        To prevent loss of data caused by 403 Error, it saves the data with pickle once in a while


        :return:
        '''


        for linkedin_url in linkedin_urls:
            try:
                with open('linkedin_difference_data', 'rb') as f:
                    data = pickle.load(f)
            except:
                data = []

            if '/company/' in linkedin_url or '/companies/' in linkedin_url: #company is temporarily not supported by the system
                continue
            data.extend(self.grabMoreLinkedInTrainingExamples(linkedin_url))

            if len(data) > 0:  # when the data is not empty
                try:
                    with open('linkedin_difference_data', 'w+b') as f:
                        pickle.dump(data, f)
                except:
                    continue
            else:
                continue
        return


    def grabText(self, webElement):
        print(type(webElement))
        if isinstance(webElement, str):
            return webElement
        elif isinstance(webElement, webdriver.remote.webelement.WebElement):
            return webElement.text.replace('\n|\t', ' ')
        else:
            return str(webElement)




'''
#test LinkedIn API
#test_linkedin_url = 'https://www.linkedin.com/in/jillian-sipkins-85234425'
test_linkedin_url = 'https://www.linkedin.com/company/patti-putnicki-freelance-writer'

soup = scrapeLinkedInPage(test_linkedin_url)



list = lkg.grabMoreLinkedInTrainingExamples('https://www.linkedin.com/in/jillian-sipkins-85234425')
print(list)
print(len(list))


'''

def main():
    train_pairs = "alta16_kbcoref_train_pairs.csv"

    ghetto_browser = webdriver.Chrome(os.getcwd() + '/chromedriver')
    ghetto_browser.get(linkedin_login_page)
    ghetto_browser.find_element_by_id('session_key-login').send_keys(linkedin_username)
    ghetto_browser.find_element_by_id('session_password-login').send_keys(linkedin_password)
    ghetto_browser.find_element_by_xpath("//form[@name='login']//input[@name='signin']").click()
    lkg = LinkedInGrab(ghetto_browser)
    #lkg.grabLinkedInDescription('https://www.linkedin.com/in/jillian-sipkins-85234425')
    linkedin_urls = lkg.harvestLinkedInLinksFromFile(train_pairs)
    lkg.harvestDeterministicLinkedInExamples(linkedin_urls)


if __name__ == "__main__":
    main()
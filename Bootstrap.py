import json
import os
import pickle
import time
from random import randint

import tweepy
from pathlib import Path

from googleapiclient.discovery import build
from selenium.webdriver import DesiredCapabilities
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from LinkedIn import grabLinkedInPage

seeds = ['potus', 'kimkardashian', 'elonmusk', 'KingJames', 'RoyalFamily' ]


def loadFromPickle(pickleFileName: str) -> object:
    with open(pickleFileName, 'rb') as f:
        try:
            object = pickle.load(f)
        except:
            object = None
    return object


def shouldSkip(url)->bool:
    skiplist = ['facebook', 'youtube']
    for domain in skiplist:
        if domain in url:
            return True
    return False



class Bootstrap:


    def __init__(self, download_new_twitter_ids):
        #Twitter API initialisation
        auth = tweepy.OAuthHandler('RhO1v4b6R2seXDgEcG3fjBBvk', 'i3ERSk3kpEnMzCTp7HzXlYucC1Vc9c6d0zzMIcaVl6Cp4J5j7y')
        auth.set_access_token('3186323958-92jMwa1haW3NYtgd6QMld0Cwvrp7txgaZC6Oe0M', '6h6M4G3ry1Y46qLuysEc5g3ll6IOsNVRwMsKLUX6CMsfw')
        self.twitter_api = tweepy.API(auth)
        #Google API initialisation

        self.google_api = build('customsearch', 'v1', developerKey = 'AIzaSyAHxNgWZufbakZBwzYO1hRKtecsP6WmRd0')


        self.pickle_file_name = 'pickledTwitterIDs'

        #If the pickled file doesn't exist, create a new one.
        if not Path(self.pickle_file_name).is_file():
            f = open(self.pickle_file_name, 'w+')
            f.close()
        else:
            #if the file is empty, it must download some new profiles
            self.twitter_profiles = loadFromPickle(self.pickle_file_name)


        #Twitter profiles are a list of tuples (twitter_username, full_name)
        if len(self.twitter_profiles)<1:
            download_new_twitter_ids = True

        if download_new_twitter_ids == True:
            self.grabPopularTwitterProfiles()

        self.grabProfiles()
        # https://developers.google.com/custom-search/ API


        return

    def grabPopularTwitterProfiles(self):
        #select seeds: eg. potus,
        for seed in seeds:

            for user in tweepy.Cursor(self.twitter_api.followers, screen_name=seed).items(300):
                with open(self.pickle_file_name,'rb') as f:
                    try:
                        self.twitter_profiles = list(pickle.load(f))
                    except tweepy.error.TweepError:
                        time.sleep(60 * 15)
                    except EOFError:
                        self.twitter_profiles = []
                initial_number = len(self.twitter_profiles)
                self.twitter_profiles.append((user.screen_name, user.name))
                print(user.screen_name)
                self.reduceDuplicateIDs()
                new_number = len(self.twitter_profiles)
                print(new_number, 'Twitter IDs collected')
                assert initial_number + 1 == new_number
                with open(self.pickle_file_name, 'w+b') as f:
                    pickle.dump(self.twitter_profiles, f)
                time.sleep(60) #Twitter API limit

        return

    def reduceDuplicateIDs(self):

        return

    #use the stored twitter API profiles and combine with Google Custom Search to retrieve summary of linkedin,

    def grabProfiles(self):

        link_description_pairs = []
        print(len(self.twitter_profiles))

        for profile in self.twitter_profiles:
            #Iteratively grab profiles
            #https://pypi.python.org/pypi/sumy  to summarise the results
            ghetto_query = '\"' + str(profile[1])+ '\"'
            link_description_pairs = self.ghetto_google_search(ghetto_query)

            #link_description_pairs = self.classy_google_search(profile[1], '012780120168009515970:hbsuzyb70xm', num=10)

        return link_description_pairs

    def getDescription(self, item):
        description =''
        if 'pagemap' in item:
            if 'metatags' in item['pagemap']:
                for metatag in item['pagemap']['metatags']:
                    if 'og:description' in metatag:
                        description = metatag['og:description'] + ' '

            if 'hcard' in item['pagemap']:
                for detail in item['pagemap']['hcard']:
                    if 'fn' in detail:
                        description += detail['fn'] + ' '

                    if 'title' in detail:
                        description += detail['title'] + ' '

            if 'person' in item['pagemap']:
                for detail in item['pagemap']['person']:
                    if 'org' in detail:
                        description += detail['org'] + ' '

                    if 'location' in detail:
                        description += detail['location']

                    if 'role' in detail:
                        description += detail['role'] + ' '

        else:
            description = item['snippet']
        return description

    def getSummaries(self):
        summaryA=''
        summaryB =''

        return summaryA, summaryB

    def classy_google_search(self, search_term, cse_id, **kwargs):
        item_description_pairs = []
        res = self.google_api.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        #print(json.dumps(res, sort_keys=True, indent=4))
        if 'items' in res:
            items = res['items']
        else:
            items = []

        for item in items:
            # archive URLs
            url = item["link"]

            description = self.getDescription(item)
            item_description_pair = (url, description)
            print(item_description_pair)
            print('#' * 20)
            item_description_pairs.append(item_description_pair)
        return item_description_pairs

    def ghetto_google_search(self, query):

        search_items = []
        ghetto_chrome = webdriver.Chrome(os.getcwd() + '/chromedriver')

        ghetto_chrome.get("http://www.google.com")
        ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        #This is so G! Credit: http://stackoverflow.com/questions/22739514/how-to-get-html-with-javascript-rendered-sourcecode-by-using-selenium

        search_box = ghetto_chrome.find_element_by_name('q')
        search_box.clear()
        search_box.send_keys(query, Keys.RETURN)
        time.sleep(randint(10,100)) #So Google doesn't block
        search_results = ghetto_chrome.find_elements_by_xpath("//*[@id='rso']")
        for search_result in search_results:
            link = search_result.find_element_by_xpath("//h3/a").get_attribute('href')

            #skip certain sites.
            if shouldSkip(str(link)):
                continue

            #handle linkedin
            if 'linkedin.com/' in link:
                description = grabLinkedInPage(ghetto_chrome)

            #handle twitter
            else:
                description = search_result.find_element_by_xpath("//span[@class='st']").text
            print(link, description)

            if not link or not description:
                continue


        ghetto_chrome.close()

        return search_items


bs = Bootstrap(download_new_twitter_ids = False)
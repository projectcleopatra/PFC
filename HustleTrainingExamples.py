import itertools
import os
import pickle
import re
import time
from random import randint

import tweepy
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.common.keys import Keys

from Utils import extractURLFeatures
from WebScraping import LinkedInGrab

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


def initialiseWebDriver():


    ghetto_browser = webdriver.Chrome(os.getcwd() + '/chromedriver')
    # linkedin login
    ghetto_browser.get(LinkedInGrab.linkedin_login_page)
    time.sleep(2)
    ghetto_browser.find_element_by_id('session_key-login').send_keys(LinkedInGrab.linkedin_username)
    ghetto_browser.find_element_by_id('session_password-login').send_keys(LinkedInGrab.linkedin_password)
    ghetto_browser.find_element_by_xpath("//form[@name='login']//input[@name='signin']").click()

    time.sleep(4)
    #facebook login
    ghetto_browser.get('https://www.facebook.com/login')
    time.sleep(2)
    ghetto_browser.find_element_by_xpath("//input[@name='email']").send_keys('passion.gratefulness.focus@gmail.com')
    ghetto_browser.find_element_by_xpath("//input[@type='password']").send_keys('e3aMpcRm')
    ghetto_browser.find_element_by_xpath("//button[@id='loginbutton']").click()
    return ghetto_browser


class Bootstrap:


    def __init__(self, download_new_twitter_ids):
        #Twitter API initialisation
        auth = tweepy.OAuthHandler('RhO1v4b6R2seXDgEcG3fjBBvk', 'i3ERSk3kpEnMzCTp7HzXlYucC1Vc9c6d0zzMIcaVl6Cp4J5j7y')
        auth.set_access_token('3186323958-92jMwa1haW3NYtgd6QMld0Cwvrp7txgaZC6Oe0M', '6h6M4G3ry1Y46qLuysEc5g3ll6IOsNVRwMsKLUX6CMsfw')
        self.twitter_api = tweepy.API(auth)
        #Google API initialisation

        self.google_api = build('customsearch', 'v1', developerKey = 'AIzaSyAHxNgWZufbakZBwzYO1hRKtecsP6WmRd0')
        self.ghetto_chrome = initialiseWebDriver()
        self.training_pairs = []


        self.lkg =  LinkedInGrab.LinkedInGrab(self.ghetto_chrome)

        # test for various social media sites
        # print(self.getYouTubeChannelProfile('https://www.youtube.com/user/jessyelmurr/featured'))
        # print(self.getGPlusProfile("https://plus.google.com/u/1/107252150757863394224"))
        # print(self.getInstagramProfile('https://www.instagram.com/mgh02/'))
        # print(self.getTwitterProfile('https://twitter.com/hernandomotta'))
        # print(self.lkg.grabLinkedInDescription('https://www.linkedin.com/in/gemmallorca'))
        # print(self.getFacebookProfile('https://facebook.com/vinovitas'))
        # print(self.getFacebookProfile('https://www.facebook.com/DoctorAngela'))
        # print(self.getPinterestProfile(('https://au.pinterest.com/pattytabachuk/')))
        # print(self.getGithubProfile('https://github.com/simsinght'))
        # print(self.getBlogspotProfile('http://bloginstructions.blogspot.com.au'))

        self.pickle_file_name = 'aboutme_examples'

        self.grabProfiles()


        # https://developers.google.com/custom-search/ API


        return

    #use the stored twitter API profiles and combine with Google Custom Search to retrieve summary of linkedin,

    def grabProfiles(self):
        '''
        Grab training profiles that are the same
        :return:
        '''



        self.ghetto_chrome.get('https://about.me/discover') #not need to be updated in the future
        self.ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        self.ghetto_chrome.implicitly_wait(8)

        #click on the load more button a few times
        load_more_button = self.ghetto_chrome.find_element_by_xpath("//*[@class='load-more']//button")

        clicks = 16 #Adjustable
        for x in range(0, clicks):
            time.sleep(6)
            load_more_button.click()

        #Acquire all profiles showed, and quietly visit all of them
        profiles = self.ghetto_chrome.find_elements_by_xpath("//li[@class='user show']")
        print(len(profiles), "profiles loaded.")

        for profile in profiles:
            self.loadTrainingPairs()
            training_pairs = []
            #Click open each profile

            descriptions = [] #IMPORTANT!  list of description of the same person from different platforms

            discovery_page = self.ghetto_chrome.current_window_handle
            self.ghetto_chrome.execute_script("arguments[0].click();", profile)

            time.sleep(6)
            #switch to new tab
            self.ghetto_chrome.switch_to.window(self.ghetto_chrome.window_handles[1])
            print('Transferred to: ', self.ghetto_chrome.current_url) #make sure we are on the right page


            #Start collecting description
            description = ''

            #name, #location, position
            name_loc_pos = self.ghetto_chrome.find_element_by_xpath("//*[@class='name-headline']").text
            description += name_loc_pos

            # bio
            bio = " ".join([paragraph.text for paragraph in self.ghetto_chrome.find_elements_by_xpath("//*[@class='bio']//div//p")])
            description += bio

            # Work Education
            work_ed_info = " ".join([item.text for item in self.ghetto_chrome.find_elements_by_xpath("//*[@class='meta']//ul//li")])
            description += work_ed_info

            #social media
            social_urls = []
            social_links = self.ghetto_chrome.find_elements_by_xpath("//*[@class='social-links']//ul//li//a")
            url_feature = ""
            for social_link in social_links:
                social_url = social_link.get_attribute('href')
                social_urls.append(social_url)
                url_feature += extractURLFeatures(social_url)

            #Give up the profile if there's no extra link shared on about.me

            #Finish up adding description for about.me profile
            description += url_feature
            descriptions.append(description)
            current_page = self.ghetto_chrome.current_window_handle

            #Go through each link
            for social_url in social_urls:

                try:
                    description = extractURLFeatures(social_url)#start with url feature

                    #snapchat is not supported
                    if 'snapchat' in social_url:
                        continue
                    elif 'youtube.com/' in social_url:
                        description += self.getYouTubeChannelProfile(social_url)
                    elif 'linkedin.com/' in social_url:
                        description += self.lkg.grabLinkedInDescription(social_url)
                        if 'url error' in description:
                            continue
                    elif 'instagram.com/' in social_url:

                        description += self.getInstagramProfile(social_url)
                    elif 'spotify' in social_url:
                        continue
                    elif 'plus.google.com' in social_url:
                        description += self.getGPlusProfile(social_url)
                    elif 'twitter.com/' in social_url:
                        description += self.getTwitterProfile(social_url)
                    elif 'facebook.com/' in social_url:
                        description += self.getFacebookProfile(social_url)
                    elif 'pinterest.com/' in social_url:
                        description +=  self.getPinterestProfile(social_url)
                    elif 'github.com/' in social_url:
                        description += self.getGithubProfile(social_url)
                    elif 'blogspot.com'in social_url:
                        description += self.getBlogspotProfile(social_url)
                    else:
                        continue

                except exceptions.NoSuchElementException:
                    print('Content not available: ', social_url)
                    continue
                description.replace('\n|\t|\r', ' ')
                descriptions.append(description)
                print('Sucessfullly scraped:', social_url)


            #Check intelligence gathering and make training examples
            print('For', name_loc_pos, len(descriptions), "profiles gathered")

            if len(descriptions)>1:
                unlabelled_training_set = list(itertools.combinations(descriptions, 2))
                for description1, description2 in unlabelled_training_set:
                    training_pairs.append((description1, description2, "1"))  # 0 when the pair refer to a different entities
                print(len(training_pairs), 'training pairs generated.s')
            #store training pairs
            self.training_pairs.extend(training_pairs)
            self.storeTrainingPairs()
            #Return to the main discovery tab
            self.ghetto_chrome.close()
            self.ghetto_chrome.switch_to.window(discovery_page)
        print('In total, ', len(self.training_pairs),'are stored.')

        return

    def ghetto_google_search(self, query):

        search_items = []
        self.ghetto_chrome = webdriver.Chrome(os.getcwd() + '/chromedriver')


        self.ghetto_chrome.get("http://www.google.com")
        self.ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        #This is so G! Credit: http://stackoverflow.com/questions/22739514/how-to-get-html-with-javascript-rendered-sourcecode-by-using-selenium

        search_box = self.ghetto_chrome.find_element_by_name('q')
        search_box.clear()
        search_box.send_keys(query, Keys.RETURN)
        time.sleep(randint(10,30)) #So Google doesn't block
        search_results = self.ghetto_chrome.find_elements_by_xpath("//*[@id='rso']")
        for search_result in search_results:
            link = search_result.find_element_by_xpath("//h3/a").get_attribute('href')

            #skip certain sites.
            if shouldSkip(str(link)):
                continue

            #handle linkedin
            if 'linkedin.com/' in link:
                description = self.lkg.grabLinkedInPage(self.ghetto_chrome)

            #handle twitter
            else:
                description = search_result.find_element_by_xpath("//span[@class='st']").text

            print('Query:', query, 'Result:', link, description)

            if not link or not description:
                continue


        self.ghetto_chrome.close()

        return search_items


    #Below are from original ideas to use the Google API.
    def getDescriptionViaGoogleAPI(self, item):
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

            description = self.getDescriptionViaGoogleAPI(item)
            item_description_pair = (url, description)
            print(item_description_pair)
            print('#' * 20)
            item_description_pairs.append(item_description_pair)
        return item_description_pairs

    def getYouTubeChannelProfile(self, social_url):
        desp = ''
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//*[contains(text(), 'About') or @aria-label = 'About tab']")
        self.ghetto_chrome.execute_script("arguments[0].click();", about)
        self.ghetto_chrome.switch_to.window(self.ghetto_chrome.current_window_handle)
        print("YouTube channel: ", self.ghetto_chrome.current_url)
        time.sleep(5)
        about_section = self.ghetto_chrome.find_element_by_xpath("//*[@class='about-metadata-container']")
        desp = ' '. join([elm.text for elm in about_section.find_elements_by_xpath(".//*") if elm.text])

        return desp

    def getInstagramProfile(self, social_url):
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//div[@class='_bugdy']")
        desp = ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])
        return desp

    def getGPlusProfile(self, social_url):
        #https://github.com/google/google-api-python-client/blob/master/samples/plus/plus.py

        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//*[contains(text(), 'About')]")
        self.ghetto_chrome.execute_script("arguments[0].click();", about)
        self.ghetto_chrome.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        time.sleep(5)
        #name and tag line

        # links - Could improve by prasing the links

        #Location

        #Work & Education
        desp = ''
        fields = self.ghetto_chrome.find_elements_by_xpath("//*[@role='region']")
        for field in fields:
            desp += ' '.join([elm.text for elm in field.find_elements_by_xpath(".//*") if elm.text])


        #find all URLs in desp and parse them into tokens
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', desp)
        #delete all urls
        desp = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', desp, flags=re.MULTILINE)
        for url in urls:
            temp_url_feature = extractURLFeatures(url)
            desp += temp_url_feature #in bag of word model, adding the urls to the end of the description doesn't change much

        return desp

    def getTwitterProfile(self, social_url):
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//*[@class='ProfileHeaderCard']")
        desp = ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])

        return desp

    def getFacebookProfile(self, social_url):
        desp=''
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        try:
            about = self.ghetto_chrome.find_element_by_xpath("//span[contains(text(), 'About')]").find_element_by_xpath("..")
            time.sleep(5)
            about.click()
            time.sleep(5)
            page_info = self.ghetto_chrome.find_elements_by_xpath("//*[@class='_4bl7']")
            for page_elm in page_info:
                desp += ' '.join([elm.text for elm in page_elm.find_elements_by_xpath(".//*") if elm.text])

        except (exceptions.NoSuchElementException, exceptions.ElementNotVisibleException):


            about = self.ghetto_chrome.find_element_by_xpath("//div//a[@data-tab-key='about' and contains(text(), 'About')]")
            about.click()
            time.sleep(2)

            #find menu items
            menu_items = self.ghetto_chrome.find_elements_by_xpath("//ul[@data-testid='info_section_left_nav']//li")
            for menu_item in menu_items:
                menu_item.click()
                time.sleep(4)
                person_info = self.ghetto_chrome.find_elements_by_xpath("//*[@class='_4bl7']")
                for page_elm in person_info:
                    self.ghetto_chrome.implicitly_wait(10)
                    desp += ' '.join([elm.text for elm in page_elm.find_elements_by_xpath(".//*") if elm.text])


        return desp

    def getPinterestProfile(self, social_url):

        desp = ''
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//div[@class='aboutBar']//div[@class='about']")
        desp += ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])
        return desp

    def getGithubProfile(self, social_url):
        desp = ''
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//*[@class='vcard-names']")
        desp += ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])
        return desp

    def storeTrainingPairs(self):
        if len(self.training_pairs)>0:
            with open(self.pickle_file_name, 'w+b') as f:
                pickle.dump(self.training_pairs, f)
        return

    def getBlogspotProfile(self, social_url):
        desp = ''
        self.ghetto_chrome.get(social_url)
        time.sleep(5)
        about = self.ghetto_chrome.find_element_by_xpath("//*[@id='header']")
        desp += ' '.join([elm.text for elm in about.find_elements_by_xpath(".//*") if elm.text])
        return desp

    def loadTrainingPairs(self):
        try:
            with open(self.pickle_file_name, 'rb') as f:
                self.training_pairs = pickle.load(f)
        except FileNotFoundError:
            self.traning_pairs = []
        return


bs = Bootstrap(download_new_twitter_ids = False)

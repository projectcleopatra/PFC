import os

import requests
import time
from bs4 import BeautifulSoup

#Instantiate the developer authentication class
from bs4 import Comment
from linkedin.linkedin import LinkedInAuthentication, PERMISSIONS, LinkedInApplication
from linkedin.server import _wait_for_user_to_enter_browser
from selenium import webdriver
#http://stackoverflow.com/questions/24880288/how-could-i-use-python-request-to-grab-a-linkedin-page
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



linkedin_username = 'passion.gratefulness.focus@gmail.com'
linkedin_password = '8YE-CXp-q8X-qCG'
linkedin_login_page = 'https://www.linkedin.com/uas/login'

def quick_api(api_key: str, secret_key: str):
    """
    This is a method copied and adapted from the original Python-LinkedIn Package --> server.py
    It automates the URL clicking

    """
    auth = LinkedInAuthentication(api_key, secret_key, 'http://localhost:8000/',
                                  list(PERMISSIONS.enums.values()))
    app = LinkedInApplication(authentication=auth)

    driver = webdriver.Chrome(os.getcwd() + '/chromedriver')
    print(auth.authorization_url)
    driver.get(auth.authorization_url)
    userName = None
    while userName == None:
        userName = driver.find_element_by_name('session_key')
        userName.send_keys(linkedin_username)
        password = driver.find_element_by_name('session_password')
        password.send_keys(linkedin_password)
    password.submit()

    _wait_for_user_to_enter_browser(app)
    driver.quit()
    return app




    #driver = webdriver.Chrome(os.getcwd()+'/chromedriver')
def useLinkedInAPI(url, application):


    if '/in/' in url:
        try:
            profile = application.get_profile(member_url=url)
        except:
            profile = ''
        print(profile)
        return profile
    elif '/companies/' in url or '/company/' in url:
        company_info = getCompanyInfo(url)
        #company API doesn't give a typical user access to company info

        return company_info
    else:
        print('New type of LinkedIn URL:', url)
        return ''

def getCompanyInfo(url):
    session = requests.session()
    login_response = session.get(linkedin_login_page)
    login = BeautifulSoup(login_response.text)

    # Get hidden form inputs
    inputs = login.find('form', {'name': 'login'}).findAll('input', {'type': ['hidden', 'submit']})

    # Create POST data
    post = {input.get('name'): input.get('value') for input in inputs}
    post['session_key'] = linkedin_username
    post['session_password'] = linkedin_password

    # Post login
    post_response = session.post('https://www.linkedin.com/uas/login-submit', data=post)

    # Get home page
    page_response = session.get(url)
    soup = BeautifulSoup(page_response.text)
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        if 'companyId' in comment:
            print(comment)

    return comment





def scrapeLinkedInPage(url: str) -> BeautifulSoup:
    '''

    :param url:
    :return:
    '''

    # Get login form

    session = requests.session()
    login_response = session.get(linkedin_login_page)
    login = BeautifulSoup(login_response.text)

    # Get hidden form inputs
    inputs = login.find('form', {'name': 'login'}).findAll('input', {'type': ['hidden', 'submit']})

    # Create POST data
    post = {input.get('name'): input.get('value') for input in inputs}
    post['session_key'] = linkedin_username
    post['session_password'] = linkedin_password

    # Post login
    post_response = session.post('https://www.linkedin.com/uas/login-submit', data=post)

    # Get home page
    page_response = session.get(url)
    soup = BeautifulSoup(page_response.text)


    return soup

class LinkedInGrab:

    def __init__(self, cookies = []):
        self.webdriver = webdriver.Chrome(os.getcwd() + '/chromedriver')
        if len(cookies)>0:
            for cookie in cookies:
                self.webdriver.add_cookie()
        self.webdriver.get(linkedin_login_page)
        self.webdriver.find_element_by_id('session_key-login').send_keys(linkedin_username)
        self.webdriver.find_element_by_id('session_password-login').send_keys(linkedin_password)
        self.webdriver.find_element_by_xpath("//form[@name='login']//input[@name='signin']").click()


    def getCookies(self):
        return self.webdriver.get_cookies()

    def grabLinkedInDescription(self, url):
        words =''
        time.sleep(3)
        self.webdriver.implicitly_wait(8)
        self.webdriver.refresh() #that causes error sometimes
        self.webdriver.get(url)
        self.webdriver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        location = self.webdriver.find_element_by_xpath("//*[@class='locality']").text
        industry = self.webdriver.find_element_by_xpath("//dd[@class='industry']//a").text
        headline = self.webdriver.find_element_by_xpath("//p[@class='title']").text
        summary =  self.webdriver.find_element_by_xpath("//div[@class='summary']//p[@class='description']").text

        education= self.webdriver.find_element_by_xpath("//tr[@id='overview-summary-education']").text
        current = self.webdriver.find_element_by_xpath("//tr[@id='overview-summary-current']").text
        previous = self.webdriver.find_element_by_xpath("//tr[@id='overview-summary-past']").text

        experience = self.webdriver.find_element_by_xpath("//div[@id='background-experience']").text
        skills = self.webdriver.find_element_by_xpath("//div[@id='background-skills']").text
        followings = self.webdriver.find_elements_by_xpath("//div[@id ='following-container']//p[@class='following-name']")
        print(headline, summary, education, current, previous, experience, skills, [following.text for following in followings])

        words = " ".join([location, industry, headline, summary, education, current, previous, experience, skills, followings]+followings)

        return words


    def grabLinkedInTrainingExamples(self, url):
        pass



def grabTwitterPage():
    pass



'''
#test LinkedIn API
#test_linkedin_url = 'https://www.linkedin.com/in/jillian-sipkins-85234425'
test_linkedin_url = 'https://www.linkedin.com/company/patti-putnicki-freelance-writer'

soup = scrapeLinkedInPage(test_linkedin_url)
'''
lkg = LinkedInGrab()
lkg.grabLinkedInDescription('https://www.linkedin.com/in/jillian-sipkins-85234425')
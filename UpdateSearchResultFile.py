import os

from selenium import webdriver

from GetData import snippet_file_schema
from Utils import csvSmartReader, csvSmartWriter
from WebScraping.LinkedInGrab import LinkedInGrab
from WebScraping.TwitterGrab import TwitterGrab

linkedin_username = 'passion.gratefulness.focus@gmail.com'
linkedin_password = '8YE-CXp-q8X-qCG'
linkedin_login_page = 'https://www.linkedin.com/uas/login'

#inistialise browser
chrome = webdriver.Chrome(os.getcwd() + '/chromedriver')


#login to linkedin
"""

chrome.get(linkedin_login_page)
chrome.find_element_by_id('session_key-login').send_keys(linkedin_username)
chrome.find_element_by_id('session_password-login').send_keys(linkedin_password)
chrome.find_element_by_xpath("//form[@name='login']//input[@name='signin']").click()
lkg = LinkedInGrab(chrome)

original = csvSmartReader("alta16_kbcoref_test_search_results.csv", snippet_file_schema)
new = csvSmartWriter("alta16_kbcoref_test_search_results.csv", snippet_file_schema)

for row in original:
    #row is a dict
    try:
        if "linkedin.com" in row['AUrl'] and "/company/" not in row["AUrl"] and "/companies/" not in row["AUrl"]:
            row['ASnippet'] = lkg.grabLinkedInDescription(row['AUrl'])
        if "linkedin.com" in row['BUrl'] and "/company/" not in row["BUrl"] and "/companies/" not in row["BUrl"]:
            row['BSnippet'] = lkg.grabLinkedInDescription(row['BUrl'])
    except:
        pass #ignore invalid URLs

    new.writerow(row)

"""
# Twitter

original = csvSmartReader("alta16_kbcoref_test_search_results.csv", snippet_file_schema)
new = csvSmartWriter("alta16_kbcoref_test_search_results", snippet_file_schema)

tg = TwitterGrab(chrome)

for row in original:
    #row is a dict
    try:
        if "twitter.com" in row['AUrl']:
            row['ASnippet'] = tg.getBio(row['AUrl'])
        if "twitter.com" in row["BUrl"]:
            row['BSnippet'] = tg.getBio(row['BUrl'])
    except:
        pass #ignore invalid URLs

    new.writerow(row)

chrome.close()



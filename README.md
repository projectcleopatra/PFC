# MakeAltaGreatAgain
This program requires full Anaconda installation with pip and Python property associated with Anaconda

## Installation

1. Create a new virtual environment when installing tensorflow
2. Make changes to the python-linkedin package so it can be compatible with python3. Details for that see Linkedin-Python Troubleshooting.md

## GetData.py
This file downloads web pages based on the links in the csv files provided. 
Downloaded pages are stored in the following data structure: 

downloaded_data = {id: dict}
dict has keys: ['id', 'A_raw_page', 'A_structured_data', 'A_last_update_datetime', 'B_raw_page',
                    'B_structured_data', 'B_last_update_datetime']

downloaded_data is stored on hard disk in the form as a pickled file. The file also exports downloaded data to a csv file                 
                    


##Typical issues:

Selenium not available --> Delete all installed Python, uninstall Anaconda --> Reinstall Anaconda
--> Doubule check pip is associated with Anaconda: which pip --> pip install Selenium


AttributeError: 'Service' object has no attribute 'process'


<<<<<<< HEAD
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def fitting(link):
    chrome_options = Options()
    chrome_options.add_argument('--kiosk')
    chrome_options.add_experimental_option('prefs', {
        'credentials_enable_service': False,
        'profile': {
            'password_manager_enabled': False
        }
    })
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    driver = webdriver.Chrome('chromedriver', options=chrome_options)
    driver.get(link)
=======
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#print cloth on monitor
def fitting(link):
    chrome_options = Options()
    chrome_options.add_argument('--kiosk')
    chrome_options.add_experimental_option('prefs', {
        'credentials_enable_service': False,
        'profile': {
            'password_manager_enabled': False
        }
    })
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    driver = webdriver.Chrome('chromedriver', options=chrome_options)
    driver.get(link)
>>>>>>> 717f58a4f915212e1b92bf7a6c3785e436c3d4d0

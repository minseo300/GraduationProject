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

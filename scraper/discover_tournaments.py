from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re

HOME_URL = 'https://www.pokedata.ovh/standingsVGC/'

options = Options()
options.add_argument("--headless")

driver = webdriver.Chrome(options=options)

tournaments = []
try:
    driver.get(HOME_URL)
    for button in driver.find_elements(By.TAG_NAME, 'button'):
        text = button.text.strip()
        onclick = button.get_attribute('onclick')
        if text and onclick:
            print(text, onclick)
            match = re.search(r"location\.href='([^']+)'", onclick)
            if match:
                tournaments.append({
                "name": text,
                "url": HOME_URL + match.group(1)
            })

finally:
    driver.quit()

print(tournaments)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re

HOME_URL = 'https://www.pokedata.ovh/standingsVGC/'

options = Options()
options.add_argument("--headless")

driver = webdriver.Chrome(options=options)

def discover_tournaments():
    tournaments = {}
    try:
        driver.get(HOME_URL)
        for button in driver.find_elements(By.TAG_NAME, 'button'):
            text = button.text.strip()
            onclick = button.get_attribute('onclick')
            if text and onclick:
                # print(text, onclick)
                match = re.search(r"location\.href='([^']+)'", onclick)
                if match:
                    tournaments[text] = {
                        'url': HOME_URL + match.group(1),
                        'id': match.group(1).replace('/', ''),
                        'divisions': []
                    }

    finally:
        pass
        # driver.quit()

    # print(tournaments)

    # the following was when tournaments was a list storing dicts of name and url
    # print(tournaments[0])
    # print(tournaments[-1])

    # first = tournaments[0]
    # recent = tournaments[-1]

    # testing to see how to pull names of divisions
    # driver.get(first['url'])

    # for button in driver.find_elements(By.TAG_NAME, 'button'):
    #     print('BUTTON:', button.text.strip(), button.get_attribute('onclick'))

    # for link in driver.find_elements(By.TAG_NAME, 'a'):
    #     print('LINK:', link.text.strip(), link.get_attribute('href'))

    # checking how to access json and csv tournament data from division page

    # divisions = ['juniors/', 'seniors/', 'masters']

    # for division in divisions:
    #     division_url = first['url'] + division
    #     print(f'\n=== {division_url} ===')

    #     driver.get(division_url)

    #     for button in driver.find_elements(By.TAG_NAME, 'button'):
    #         print('BUTTON:', button.text.strip(), button.get_attribute('onclick'))

    #     for link in driver.find_elements(By.TAG_NAME, 'a'):
    #         print('LINK:', link.text.strip(), link.get_attribute('href'))

    # source = driver.page_source
    # index = source.lower().find("json")

    # if index != -1:
    #     print(source[index-500:index+500])

    # add division names and links to tournament dictionary
    for name, tournament in tournaments.items():
        driver.get(tournament['url'])

        for button in driver.find_elements(By.TAG_NAME, 'button'):
            onclick = button.get_attribute('onclick')
            match = re.search(r"location\.href='([^']+)'", onclick)
            text = button.text.strip()

            if match and text:
                tournament['divisions'].append({
                    'name': text,
                    'url': tournament['url'] + match.group(1)
                })

    driver.quit()

    # print(tournaments)

    return tournaments
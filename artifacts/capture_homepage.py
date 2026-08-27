from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

root = Path(__file__).resolve().parent
options = Options()
options.add_argument('--headless=new')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--hide-scrollbars')
driver = webdriver.Chrome(options=options)
try:
    for name, width, height in [('homepage-desktop.png', 1440, 1000), ('homepage-mobile.png', 390, 844)]:
        driver.set_window_size(width, height)
        driver.get('http://127.0.0.1:5000/')
        WebDriverWait(driver, 10).until(lambda d: d.find_element(By.ID, 'translator-workspace'))
        page_height = driver.execute_script('return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)')
        driver.set_window_size(width, page_height)
        driver.save_screenshot(str(root / name))
finally:
    driver.quit()

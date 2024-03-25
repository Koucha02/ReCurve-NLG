from selenium import webdriver
proxy="127.0.0.1:8888"
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")#设置无头模式
chrome_options.add_argument('--proxy-server={}'.format(proxy))#设置代理
driver=webdriver.Chrome("./chromedriver", chrome_options=chrome_options)
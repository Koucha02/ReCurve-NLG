from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import os

def translate_images(input_folder, output_folder):
    # 设置Chrome WebDriver路径
    chrome_driver_path = r'E:\PycharmPrograms\Recurve\msedgedriver.exe'  # 替换为你的ChromeDriver路径
    service = Service(chrome_driver_path)
    browser = webdriver.Chrome(service=service)
    browser.get("https://translate.google.com/?sl=ja&tl=zh-CN&op=images")
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            input_image_path = os.path.join(input_folder, filename)
            # 打开Google翻译图片页面
            # 上传图片
            # time.sleep(2)
            upload_button = browser.find_element("xpath", "//span[contains(text(),'浏览文件')]/parent::div/input")
            upload_button.send_keys(input_image_path)

            # 等待上传完成
            time.sleep(5)

            # 等待翻译完成
            while True:
                try:
                    download_button = browser.find_element("xpath", "//span[contains(text(),'下载译文')]")
                    break
                except:
                    time.sleep(2)

            # 点击下载译文按钮
            download_button.click()

            # 等待下载完成
            time.sleep(5)

            # 保存翻译后的图片
            output_image_path = os.path.join(output_folder, f"translated_{filename}")
            browser.save_screenshot(output_image_path)

            print(f"翻译完成！翻译图片已保存到: {output_image_path}")

    # 关闭浏览器
    browser.quit()

if __name__ == "__main__":
    input_folder = r"C:\Users\46959\Downloads\HM_Story\converted_images"  # 替换为你的输入文件夹路径
    output_folder = r"C:\Users\46959\Downloads\HM_Story\translated"  # 替换为你的输出文件夹路径

    translate_images(input_folder, output_folder)

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Chrome 드라이버 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않고 실행
chrome_options.add_argument("--no-sandbox")  # 샌드박스 비활성화 (Linux 환경)
chrome_options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 방지

def get_driver(output_dir):
    options = webdriver.ChromeOptions()

    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--headless")  # 헤드리스 모드를 활성화하여 브라우저 UI를 표시하지 않고 백그라운드에서 실행 / 브라우저 실행 과정을 직접 확인하고 싶으시다면, 주석을 제거해 주세요.
    options.add_argument("--disable-gpu")  # GPU를 비활성화
    options.add_argument("--no-sandbox")  # 브라우저를 제한된 권한으로 실행하는 경우 샌드박스 모드를 비활성화 (Colab에서 필수)

    options.add_experimental_option('prefs', {
        "download.default_directory": output_dir, #Change default directory for downloads
        "download.prompt_for_download": False, #To auto download the file
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True #It will not show PDF directly in chrome
        })

    driver = webdriver.Chrome(options=options)

    return driver
  
  
driver = get_driver('.')

# 웹 페이지 열기
url = "https://www.kyochul.com"
driver.get(url)

# 동적 콘텐츠가 로드될 때까지 기다림
time.sleep(5)  # 충분한 대기 시간을 설정 (필요시 WebDriverWait 사용)

# 렌더링된 HTML 가져오기
html_source = driver.page_source

# HTML 파일로 저장
with open("kyochul_dynamic.html", "w", encoding="utf-8") as file:
    file.write(html_source)

# 드라이버 종료
driver.quit()

print("동적 HTML 코드가 저장되었습니다.")
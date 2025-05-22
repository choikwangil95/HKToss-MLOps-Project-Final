from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 5, 13),
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

dag = DAG(
    dag_id="rss_news_monitor",
    default_args=default_args,
    schedule_interval="* * * * *",  # 매 1분
    catchup=False,
    description="연합뉴스 RSS에서 새 뉴스 수집",
)

NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"
LAST_CRAWLED_FILE = "/opt/airflow/last_crawled.txt"


def fetch_article_details(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.google.com",
    }

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    # 대표 이미지
    image = soup.select_one('meta[property="og:image"]')["content"]

    # 기사 본문 (HTML 태그 제거한 텍스트)
    article = soup.select_one("article#dic_area").get_text(strip=True, separator="\n")

    return image, article


def convert_to_public_url(href):

    parsed = urlparse(href)
    params = parse_qs(parsed.query)
    article_id = params.get("article_id", [""])[0]
    office_id = params.get("office_id", [""])[0]
    if article_id and office_id:
        return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"

    return href


def fetch_latest_news():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",  # Do Not Track
        "Upgrade-Insecure-Requests": "1",
    }

    res = requests.get(NEWS_URL, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    # 마지막 크롤링 시각 읽기
    last_time = None
    if os.path.exists(LAST_CRAWLED_FILE):
        with open(LAST_CRAWLED_FILE, "r") as f:
            last_time = f.read().strip()

    articles = soup.select("dl > dd.articleSummary")
    new_articles = []

    for article in articles:
        title_tag = article.find_previous_sibling("dd", class_="articleSubject").a
        title = title_tag.text.strip()
        url = convert_to_public_url(title_tag["href"])
        press = article.select_one(".press").text.strip()
        wdate = article.select_one(".wdate").text.strip()  # 예: 2025-05-22 11:34

        # 날짜 비교
        if last_time is None or wdate > last_time:
            new_articles.append(
                {"title": title, "url": url, "press": press, "wdate": wdate}
            )

    # 새 뉴스가 있다면 저장하거나 로깅
    if new_articles:
        for article in new_articles:
            # 기사 본문과 이미지를 가져옴
            image, article_text = fetch_article_details(article["url"])

            print(
                f"[NEW] {article['wdate']} - {article['title']} ({article['press']}) - {article['url']}"
            )
            print(f"{article_text}\n")
        # 최신 뉴스 기준으로 last_time 갱신
        latest_time = max(article["wdate"] for article in new_articles)
        with open(LAST_CRAWLED_FILE, "w") as f:
            f.write(latest_time)
    else:
        print("\n-----------------------새 뉴스 없음!-----------------------\n")

    return new_articles


check_news_task = PythonOperator(
    task_id="check_rss_news",
    python_callable=fetch_latest_news,
    dag=dag,
)

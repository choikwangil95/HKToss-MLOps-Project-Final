from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from bs4 import SoupStrainer
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import time
# import redis

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 5, 13),
    "retries": 0,
    # "retry_delay": timedelta(seconds=30),
}

dag = DAG(
    dag_id="rss_news_monitor",
    default_args=default_args,
    schedule_interval="* * * * *",  # 매 1분
    catchup=False,
    description="네이버페이 증권 종목 뉴스에서 새 뉴스 수집",
)

NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"
LAST_CRAWLED_FILE = "/tmp/last_crawled.txt"


# def publish_to_redis(channel, message):
#     r = redis.Redis(
#         host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", 6379))
#     )
#     r.publish(channel, message)


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

def parse_wdate(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M")

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

    print(f"🧪 last_time: {last_time}")

    articles = soup.select("dl > dd.articleSummary")
    new_articles = []

    for article in articles:
        title_tag = article.find_previous_sibling("dd", class_="articleSubject").a
        title = title_tag.text.strip()
        url = convert_to_public_url(title_tag["href"])
        press = article.select_one(".press").text.strip()
        wdate = article.select_one(".wdate").text.strip()  # 예: 2025-05-22 11:34

        # 날짜 비교
        last_time_dt = parse_wdate(last_time) if last_time else None
        article_time_dt = parse_wdate(wdate)

        if last_time_dt is None or article_time_dt > last_time_dt:
            new_articles.append(
                {"title": title, "url": url, "press": press, "wdate": wdate}
            )

    print(f"🧪 수집된 새 뉴스 개수: {len(new_articles)}")

    print(f"🧪 new_articles 내용: {new_articles}")
    print(f"🧪 각 기사 wdate: {[article['wdate'] for article in new_articles]}")


    # 새 뉴스가 있다면 저장하거나 로깅
    if new_articles:
        for article in new_articles[:5]:
            try:
                image, article_text = fetch_article_details(article["url"])
                print(
                    f"[NEW] {article['wdate']} - {article['title']} ({article['press']}) - {article['url']}"
                )
                print(f"{article_text[:300]}...\n")  # 너무 긴 경우 생략
            except Exception as e:
                print(f"❌ 기사 내용 파싱 실패: {e}")
                continue

        try:
            latest_time = max(parse_wdate(article["wdate"]) for article in new_articles)
            if not os.path.exists(os.path.dirname(LAST_CRAWLED_FILE)):
                os.makedirs(os.path.dirname(LAST_CRAWLED_FILE), exist_ok=True)

            print(f"🧪 last_time: {last_time}")
            print(f"🧪 최신 뉴스 시간: {latest_time}")

            with open(LAST_CRAWLED_FILE, "w") as f:
                f.write(latest_time.strftime("%Y-%m-%d %H:%M"))
        except Exception as e:
            print(f"❌ 마지막 시간 기록 실패: {e}")
    else:
        print("\n-----------------------새 뉴스 없음!-----------------------\n")

    return new_articles


check_news_task = PythonOperator(
    task_id="check_rss_news",
    python_callable=fetch_latest_news,
    dag=dag,
)

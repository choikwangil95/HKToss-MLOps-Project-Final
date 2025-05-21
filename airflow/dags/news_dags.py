from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import os
import time

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 13),
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

dag = DAG(
    dag_id='rss_news_monitor',
    default_args=default_args,
    schedule_interval='* * * * *',  # 매 1분
    catchup=False,
    description='연합뉴스 RSS에서 새 뉴스 수집',
)

RSS_URL = "https://www.yna.co.kr/rss/economy.xml"
STATE_FILE = "/opt/airflow/rss_latest_timestamp.txt"

def extract_yna_article_body(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except Exception as e:
        return f"[에러] 요청 실패: {e}"

    only_article = SoupStrainer("div", class_="story-news article")
    soup = BeautifulSoup(res.text, "lxml", parse_only=only_article)

    content_div = soup.select_one("div.story-news.article")
    if not content_div:
        return "[본문 없음]"
    
    paragraphs = content_div.find_all("p")

    body = "\n".join([
        p.get_text(strip=True)
        for p in paragraphs if p.get_text(strip=True) and not p.get_text(strip=True).startswith("©")
    ])

    return body

def check_new_rss_news():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                latest_ts_str = f.read().strip()
                latest_published = time.strptime(latest_ts_str, "%a, %d %b %Y %H:%M:%S %z")
        else:
            latest_published = None
    except Exception:
        latest_published = None

    feed = feedparser.parse(RSS_URL)
    new_articles = []

    for entry in feed.entries:
        if latest_published is None or entry.published_parsed > latest_published:
            new_articles.append(entry)

    if new_articles:
        latest_ts = new_articles[0].published
        with open(STATE_FILE, 'w') as f:
            f.write(latest_ts)

        for article in new_articles:
            print(f"[{article.published}] {article.title} - {article.link}")

            body = extract_yna_article_body(article.link)

            print(f"\n--- 본문 ---\n{body}\n")  # 본문 앞부분만 출력
    else:
        print("새 뉴스 없음.")

check_news_task = PythonOperator(
    task_id='check_rss_news',
    python_callable=check_new_rss_news,
    dag=dag,
)

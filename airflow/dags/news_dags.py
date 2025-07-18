from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
from bs4 import SoupStrainer
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import os
import random
import time
import concurrent.futures
import logging
from requests.adapters import HTTPAdapter, Retry

log = logging.getLogger(__name__)

NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"
LAST_CRAWLED_FILE = "/opt/airflow/data/last_crawled.txt"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 5, 13),
    "retries": 0,
}

dag = DAG(
    dag_id="rss_news_monitor",
    default_args=default_args,
    schedule_interval="* * * * *",
    catchup=False,
    description="네이버페이 증권 종목 뉴스에서 새 뉴스 수집",
)

# ──────────────────────────────
# 📌 유틸 함수
# ──────────────────────────────


def parse_wdate(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def convert_to_public_url(href):
    parsed = urlparse(href)
    params = parse_qs(parsed.query)
    article_id = params.get("article_id", [""])[0]
    office_id = params.get("office_id", [""])[0]
    if article_id and office_id:
        return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"
    return href


def get_random_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }


def safe_soup_parse(text, timeout=3):
    def parse():
        return BeautifulSoup(text, "lxml")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(parse)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.error("❌ soup 파싱 타임아웃 발생")
            return None
        except Exception as e:
            log.error(f"❌ soup 파싱 실패: {type(e).__name__}: {e}")
            return None


def get_retry_session():
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_article_details(url):
    try:
        headers = get_random_headers()
        log.info(f"📰 요청 URL: {url}")

        session = get_retry_session()
        res = session.get(url, headers=headers, timeout=10)
        log.info(f"📅 응답 상태 코드: {res.status_code}")
        res.raise_for_status()
        log.info(f"📄 응답 본문 길이: {len(res.text)}")

        soup = safe_soup_parse(res.text)
        if soup is None:
            return None, ""

        log.info("✅ soup 생성 완료")

        image_tag = soup.select_one('meta[property="og:image"]')
        image = (
            image_tag["content"]
            if image_tag and image_tag.has_attr("content")
            else None
        )

        article_tag = soup.select_one("article#dic_area")
        article = (
            article_tag.get_text(strip=True, separator="\n") if article_tag else ""
        )

        log.info(f"✅ 추출 성공: 이미지 있음? {bool(image)}, 본문 길이: {len(article)}")

        # 요청 사이에 무작위 대기
        time.sleep(random.uniform(1.0, 2.5))

        return image, article

    except Exception as e:
        log.error(f"❌ 전체 fetch 실패 - {url}: {type(e).__name__}: {e}")
        return None, ""


def get_or_create_last_time(filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                last_time = f.read().strip()
            if last_time:
                log.info(f"🧪 이전 기록된 시간: {last_time}")
                return last_time
        log.info("📁 기록이 없어 현재 시각으로 초기화")
        with open(filepath, "w") as f:
            f.write(now_str)
        return now_str
    except Exception as e:
        log.error(f"❌ 시간 읽기/쓰기 실패 ({type(e).__name__}): {e}")
        return now_str


def save_latest_time(filepath: str, time_str: str):
    try:
        with open(filepath, "w") as f:
            f.write(time_str)
        log.info(f"✅ 최신 시간 저장: {time_str}")
    except Exception as e:
        log.error(f"❌ 시간 저장 실패 ({type(e).__name__}): {e}")


# ──────────────────────────────
# 📌 뉴스 수집 메인 함수
# ──────────────────────────────


def fetch_latest_news():
    try:
        res = requests.get(NEWS_URL, headers=get_random_headers(), timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
    except Exception as e:
        log.error(f"❌ 뉴스 목록 요청 실패 ({type(e).__name__}): {e}")
        return []

    last_time = get_or_create_last_time(LAST_CRAWLED_FILE)
    last_time_dt = parse_wdate(last_time)

    articles = soup.select("dl > dd.articleSummary")
    new_articles = []

    for article in articles:
        try:
            title_tag = article.find_previous_sibling("dd", class_="articleSubject").a
            title = title_tag.text.strip()
            url = convert_to_public_url(title_tag["href"])
            press = article.select_one(".press").text.strip()
            wdate = article.select_one(".wdate").text.strip()
            article_time_dt = parse_wdate(wdate)

            if article_time_dt > last_time_dt:
                new_articles.append(
                    {"title": title, "url": url, "press": press, "wdate": wdate}
                )
        except Exception as e:
            log.error(f"❌ 개별 뉴스 파싱 실패 ({type(e).__name__}): {e}")

    log.info(f"🧪 새 뉴스 수: {len(new_articles)}")

    if new_articles:
        latest_time = max(parse_wdate(a["wdate"]) for a in new_articles)
        save_latest_time(LAST_CRAWLED_FILE, latest_time.strftime("%Y-%m-%d %H:%M"))

        for article in new_articles[:5]:
            try:
                log.info(f"\n📰 기사 처리 중: {article['title']}")
                image, article_text = fetch_article_details(article["url"])
                preview = article_text[:300] if isinstance(article_text, str) else ""
                log.info(
                    f"[NEW] {article['wdate']} - {article['title']} ({article['press']})"
                )
                log.info(f"{preview}...\n")
            except Exception as e:
                log.error(f"❌ 본문 파싱 실패 ({type(e).__name__}): {e}")
    else:
        log.info("⏰ 새 뉴스 없음")

    return new_articles


# ──────────────────────────────
# 📌 DAG 등록
# ──────────────────────────────

check_news_task = PythonOperator(
    task_id="check_rss_news",
    python_callable=fetch_latest_news,
    dag=dag,
)

from news_pipeline import fetch_latest_news
import schedule
import time
import logging

log = logging.getLogger("news_logger")

def job():
    log.info("🕒 [스케줄러] 뉴스 수집 실행")

    # 1 뉴스 실시간 수집 실행 함수
    news_crawled = fetch_latest_news()

    # 2 뉴스 본문 전처리 실행 함수

    # 3 뉴스 본문 요악 함수

    # 4 

    # 5 

if __name__ == "__main__":
    log.info("✅ run_scheduler.py 시작됨")

    # 🟡 첫 실행 즉시
    job()

    # 🟢 이후 매 1분마다 실행
    schedule.every(1).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)

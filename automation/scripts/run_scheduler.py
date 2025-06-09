from news_pipeline import fetch_latest_news, remove_market_related_sentences, get_summarize_model, summarize_event_focused, get_ner_pipeline, get_stock_names, load_official_stock_list, filter_official_stocks_from_list, load_stock_to_industry_map, get_industry_list_from_stocks, save_to_db_metadata
import schedule
import time
import logging
import os
import joblib

log = logging.getLogger("news_logger")


def job(tokenizer_summarize, model_summarize, device, ner_pipe, official_stock_set, stock_to_industry, vectorizer, lda_model, stopwords):
    log.info("🕒 [스케줄러] 뉴스 수집 실행")

    # ──────────────────────────────
    # 1 뉴스 실시간 수집
    # - 1분마다 수집
    # ──────────────────────────────

    # 1 뉴스 실시간 수집 실행 함수
    news_crawled = fetch_latest_news()

    # ──────────────────────────────
    # 2 뉴스 전처리
    # - 본문 전처리 및 요약, 종목과 업종명 매칭
    # ──────────────────────────────

    # 1 뉴스 본문 전처리 실행 함수
    filtered_news = []

    if len(news_crawled) != 0:
        for news in news_crawled:
            news_article = news["article"]
            news_article_preprocessed = remove_market_related_sentences(news_article)

            if len(news_article_preprocessed) < 70:
                continue  # 본문 길이 짧으면 제외

            news["article_preprocessed"] = news_article_preprocessed
            filtered_news.append(news)

    print(f"\n필터린됭 뉴스 {filtered_news}\n")

    # 2 뉴스 본문 요악 함수
    summarzied_news = []

    if len(filtered_news) != 0:
        for news in filtered_news:
            news_article = news["article_preprocessed"]
            news_article_summarized = summarize_event_focused(news_article, tokenizer_summarize, model_summarize, device)

            if len(news_article_summarized) < 70:
                continue  # 본문 길이 짧으면 제외

            news["summary"] = news_article_summarized
            summarzied_news.append(news)

    print(f"\n요약 뉴스 확인 {summarzied_news}\n")

    ner_news = []

    # 3 뉴스 종목, 업종명 매칭 함수
    if len(summarzied_news) != 0:
        for news in summarzied_news:
            news_summary = news["summary"]
            stock_list = get_stock_names(ner_pipe, news_summary)

            # 🔽 여기서 필터링
            stock_list = filter_official_stocks_from_list(stock_list, official_stock_set)

            if len(stock_list) > 4 or len(stock_list) < 1:
                continue  # 종목 없거나 너무 많으면 제외

            industry_list = get_industry_list_from_stocks(stock_list, stock_to_industry)

            if len(industry_list) < 1:
                continue

            news["stock_list"] = stock_list
            news["industry_list"] = industry_list

            ner_news.append(news)

    print(f"\n종목, 업종명 매칭 뉴스 확인 {ner_news}\n")

    save_to_db_metadata(ner_news)


# ──────────────────────────────
# 3 뉴스 경제 및 행동 지표 피쳐 추가
# - 주가 D+1~D+30 변동률, 금리, 환율, 기관 매매동향, 유가 등
# ──────────────────────────────

# 으악

# ──────────────────────────────
# 4 뉴스 시멘틱 피쳐 추가
# - topic별 분포값, 클러스터 동일 여부
# ──────────────────────────────

# 으악


if __name__ == "__main__":
    log.info("🟡 summarize 모델 불러오는 중...")
    tokenizer_summarize, model_summarize, device = get_summarize_model()
    log.info("🟢 summarize 모델 로딩 완료")

    log.info("🟡 NER 모델 불러오는 중...")
    ner_pipe = get_ner_pipeline()
    log.info("🟢 NER 모델 로딩 완료")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    official_stock_path = os.path.join(BASE_DIR, "../db/KRX_KOSPI.csv")
    industry_map_path = os.path.join(BASE_DIR, "../db/kospi_description.csv")

    official_stock_set = load_official_stock_list(official_stock_path)
    stock_to_industry = load_stock_to_industry_map(industry_map_path)

    lda_model_path = os.path.join(BASE_DIR, "../db/best_lda_model.pkl")
    count_vectorizer_path = os.path.join(BASE_DIR, "../db/count_vectorizer.pkl")
    stopwords_path = os.path.join(BASE_DIR, "../db/stopwords-ko.txt")

    vectorizer = joblib.load(count_vectorizer_path)
    lda_model = joblib.load(lda_model_path)
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]

    log.info("✅ run_scheduler.py 시작됨")

    # 첫 실행 즉시
    job(tokenizer_summarize, model_summarize, device, ner_pipe, official_stock_set, stock_to_industry, vectorizer, lda_model, stopwords)

    # 이후 매 1분마다 실행
    schedule.every(1).minutes.do(lambda: job(tokenizer_summarize, model_summarize, ner_pipe, device, official_stock_set, stock_to_industry, vectorizer, lda_model, stopwords))

    while True:
        schedule.run_pending()
        time.sleep(1)

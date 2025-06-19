import numpy as np
import re
from konlpy.tag import Okt
import numpy as np
from sqlalchemy.orm import Session

from schemas.model import SimilarNewsItem
import asyncio
from fastapi.responses import StreamingResponse
import json
import redis
from concurrent.futures import ThreadPoolExecutor

import ast
import pandas as pd


def get_news_summary(
    text,
    request,
    max_length=128,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
) -> str:
    """
    뉴스 본문을 요약하는 함수입니다.
    실제 요약 로직은 외부 모델이나 라이브러리를 사용하여 구현해야 합니다.
    """
    encoder_sess = request.app.state.encoder_sess_summarize
    decoder_sess = request.app.state.decoder_sess_summarize
    tokenizer = request.app.state.tokenizer_summarize

    text = text.strip()[:300]

    input_ids = tokenizer.encode(text).ids
    input_ids_np = np.array([input_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids_np, dtype=np.int64)

    encoder_outputs = encoder_sess.run(
        None, {"input_ids": input_ids_np, "attention_mask": attention_mask}
    )[0]

    decoder_input_ids = [tokenizer.token_to_id("<s>")]
    generated_ids = decoder_input_ids.copy()

    for _ in range(max_length):
        decoder_input_np = np.array([generated_ids], dtype=np.int64)
        decoder_inputs = {
            "input_ids": decoder_input_np,
            "encoder_hidden_states": encoder_outputs,
            "encoder_attention_mask": attention_mask,
        }
        logits = decoder_sess.run(None, decoder_inputs)[0]
        next_token_logits = logits[:, -1, :]

        # repetition penalty 적용
        for token_id in set(generated_ids):
            next_token_logits[0, token_id] /= repetition_penalty

        # no_repeat_ngram_size 적용
        if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size:
            ngram = tuple(generated_ids[-(no_repeat_ngram_size - 1) :])
            banned = {
                tuple(generated_ids[i : i + no_repeat_ngram_size])
                for i in range(len(generated_ids) - no_repeat_ngram_size + 1)
            }
            for token_id in range(next_token_logits.shape[-1]):
                if ngram + (token_id,) in banned:
                    next_token_logits[0, token_id] = -1e9  # 큰 마이너스

        # greedy 선택
        next_token_id = int(np.argmax(next_token_logits, axis=-1)[0])

        if next_token_id == tokenizer.token_to_id("</s>"):
            break

        generated_ids.append(next_token_id)

    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return summary


def get_ner_tokens(text, request, id2label):
    """
    뉴스 본문에서 개체명 인식을 수행하는 함수입니다.
    """
    tokenizer = request.app.state.tokenizer_ner
    session = request.app.state.session_ner

    # 🟡 토큰화 및 입력값 준비
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    # 🔵 ONNX 추론 실행
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    logits = session.run(None, inputs)[0]  # shape: (1, seq_len, num_labels)

    # 🔵 라벨 인덱스 → 실제 라벨명
    preds = np.argmax(logits, axis=-1)[0]
    labels = [id2label[p] for p in preds[: len(encoding.tokens)]]

    # 🔵 시각화
    tokens = encoding.tokens

    return tokens, labels


def extract_ogg_economy(tokens, labels, target_label="OGG_ECONOMY"):
    merged_words = []
    current_word = ""

    for token, label in zip(tokens, labels):
        token_clean = token.replace("##", "") if token.startswith("##") else token

        if label == f"B-{target_label}":
            if current_word:
                merged_words.append(current_word)
            current_word = token_clean

        elif label == f"I-{target_label}":
            current_word += token_clean

        else:
            if current_word:
                merged_words.append(current_word)
                current_word = ""

    if current_word:
        merged_words.append(current_word)

    stock_list = merged_words.copy()

    return stock_list


async def get_news_embeddings(article_list, request):
    """
    뉴스 본문 리스트를 임베딩하는 함수입니다.
    ONNX 모델이 배치 입력을 지원할 경우, 한 번에 추론합니다.
    """
    tokenizer = request.app.state.tokenizer_embedding
    session = request.app.state.session_embedding

    # 1. 토큰화
    encoded = [tokenizer.encode(x) for x in article_list]
    input_ids = [e.ids for e in encoded]
    attention_mask = [[1] * len(ids) for ids in input_ids]

    # 2. 패딩 (최대 길이 기준)
    max_len = max(len(ids) for ids in input_ids)
    input_ids_padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
    attention_mask_padded = [
        mask + [0] * (max_len - len(mask)) for mask in attention_mask
    ]

    # 3. numpy 배열로 변환
    input_ids_np = np.array(input_ids_padded, dtype=np.int64)
    attention_mask_np = np.array(attention_mask_padded, dtype=np.int64)

    # 4. ONNX 추론
    outputs = session.run(
        ["sentence_embedding"],
        {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
    )[
        0
    ]  # shape: (batch_size, hidden_dim)

    # 5. 반환 (List[List[float]])
    return outputs.tolist()


def safe_parse_list(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []


def get_news_similar_list(payload, request):
    """
    유사 뉴스 top_k
    """
    article = payload.article
    top_k = payload.top_k

    vectordb = request.app.state.vectordb

    # 검색
    results = vectordb.similarity_search_with_score(article, k=100)

    news_similar_list = []
    seen_titles = set()

    for doc, score in results:
        similarity = round(1 - float(score), 2)

        if similarity > 0.9:
            continue  # 유사도가 너무 높으면 제외 (0.9 이상 필터링)

        title = doc.metadata.get("title")
        if title in seen_titles:
            continue  # 이미 추가한 title이면 스킵
        seen_titles.add(title)

        news_id = doc.metadata.get("news_id")
        wdate = doc.metadata.get("wdate")
        summary = doc.page_content
        url = doc.metadata.get("url")
        image = doc.metadata.get("image")
        stock_list = safe_parse_list(doc.metadata.get("stock_list"))
        industry_list = safe_parse_list(doc.metadata.get("industry_list"))

        if not news_id or not wdate or not summary:
            continue

        news_similar_list.append(
            SimilarNewsItem(
                news_id=news_id,
                wdate=wdate,
                title=title,
                summary=summary,
                url=url,
                image=image,
                stock_list=stock_list,
                industry_list=industry_list,
                similarity=similarity,
            )
        )

    return news_similar_list[:top_k]


def get_lda_topic(text, request):
    lda_model = request.app.state.lda_model
    count_vectorizer = request.app.state.count_vectorizer
    stopwords = request.app.state.stopwords

    # 2. 형태소 분석기 및 불용어 로드
    okt = Okt()

    # 3. 텍스트 정제 함수
    def clean_text(text):
        text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
        text = re.sub(r"[^가-힣\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # 4. 명사 추출 함수
    def extract_nouns(text):
        nouns = okt.nouns(text)
        nouns = [word for word in nouns if word not in stopwords and len(word) > 1]
        return " ".join(nouns)

    # 5. 데이터 전처리 (정제 + 명사 추출)
    processed_texts = [extract_nouns(clean_text(text))]

    # 6. 벡터라이즈 (DTM 생성)
    new_dtm = count_vectorizer.transform(processed_texts)

    # 7. LDA 토픽 분포 예측
    topic_distribution = lda_model.transform(new_dtm)

    lda_topics = {}
    for index, value in enumerate(topic_distribution[0]):
        lda_topics[f"topic_{index+1}"] = value

    return lda_topics


# ✅ 전역 Redis 연결 재사용
redis_conn = redis.Redis(
    host="43.200.17.139",
    port=6379,
    password="q1w2e3r4!@#",
    decode_responses=True,
)

# ✅ 전역 ThreadPoolExecutor (스레드 재사용)
redis_executor = ThreadPoolExecutor(max_workers=10)


# ✅ Redis 비동기 전송 함수
def send_to_redis_async(data: dict):
    def task():
        try:
            redis_conn.publish("chat-response", json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"[Redis Error] {type(e).__name__}: {e}")

    redis_executor.submit(task)


# ✅ SSE 응답
async def get_stream_response(request, payload):
    chatbot = request.app.state.chatbot
    loop = asyncio.get_event_loop()

    # 프롬프트 생성 (동기 → 비동기)
    prompt = await loop.run_in_executor(
        None, chatbot.make_stream_prompt, payload.question, 5
    )

    client = chatbot.get_client()
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1024,
        stream=True,
    )

    queue = asyncio.Queue()

    def lookahead_iter(iterator):
        """마지막 여부를 알려주는 제너레이터 (yield (is_last, item))"""
        it = iter(iterator)
        try:
            prev = next(it)
        except StopIteration:
            return

        for val in it:
            yield False, prev
            prev = val
        yield True, prev  # 마지막

    last_sent = False  # 마지막 true가 나갔는지 추적

    def produce_chunks():
        nonlocal last_sent
        idx = 0  # 인덱스 추가
        try:
            for is_last, chunk in lookahead_iter(stream):
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)

                if content:
                    data = {
                        "client_id": payload.client_id,
                        "content": content,
                        "is_last": is_last,
                        "index": idx,  # 인덱스 부여
                    }
                    idx += 1  # 인덱스 증가
                    if is_last:
                        last_sent = True

                    send_to_redis_async(data)
                    msg = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
        finally:
            # 혹시 마지막 is_last가 안 나갔으면 강제로 전송
            if not last_sent:
                data = {
                    "client_id": payload.client_id,
                    "content": "",  # 또는 None
                    "is_last": True,
                    "index": idx,  # 마지막 인덱스
                }
                send_to_redis_async(data)
                msg = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    # 백그라운드로 실행
    loop.run_in_executor(None, produce_chunks)

    # 비동기 스트림
    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# AE 인코딩 함수
def run_ae(ae_sess, embedding):
    input_name = ae_sess.get_inputs()[0].name
    output_name = ae_sess.get_outputs()[0].name
    return ae_sess.run([output_name], {input_name: embedding.astype(np.float32)})[0]


# 그룹 단위 스케일링 함수 (그룹별 scaler 적용)
def scale_ext_grouped(
    ext: list, col_names: list, prefix: str, scalers: dict, group_key_map: dict
):
    grouped_data = {}
    grouped_indices = {}
    for idx, (col, val) in enumerate(zip(col_names, ext)):
        group = group_key_map.get(col, None)
        if group:
            key = f"{prefix}_{group}"
            grouped_data.setdefault(key, []).append(val)
            grouped_indices.setdefault(key, []).append(idx)

    scaled = ext.copy()
    for key in grouped_data:
        if key in scalers:
            try:
                values = np.array(grouped_data[key], dtype=np.float32).reshape(1, -1)
                # transformed = scalers[key].transform(values)[0]

                columns = scalers[key].feature_names_in_  # sklearn >=1.0
                values_df = pd.DataFrame(values, columns=columns)
                transformed = scalers[key].transform(values_df)[0]

                for idx, val in zip(grouped_indices[key], transformed):
                    scaled[idx] = val
            except Exception as e:
                print(f"❌ {key} 스케일 실패: {e}")
                raise
        else:
            print(f"⚠️ {key} 스케일러 없음 → 원본 사용")

    return np.array(scaled, dtype=np.float32)


# 회귀 기반 유사 뉴스 유사도 계산 함수
async def compute_similarity(
    db: Session,
    summary: str,
    extA: list,
    topicA: list,
    similar_summaries: list,
    extBs: list,
    topicBs: list,
    scalers,
    ae_sess,
    regressor_sess,
    embedding_api_func,
    ext_col_names: list,
    topic_col_names: list,
    news_topk_ids: list,
):

    # group_key_map 생성 (기준 + 유사 뉴스)
    group_key_map = {}
    for col in ext_col_names + topic_col_names:
        if "date_close" in col:
            group_key_map[col] = "price_close"
        elif "date_volume" in col:
            group_key_map[col] = "volume"
        elif "date_foreign" in col:
            group_key_map[col] = "foreign"
        elif "date_institution" in col:
            group_key_map[col] = "institution"
        elif "date_individual" in col:
            group_key_map[col] = "individual"
        elif col in ["fx", "bond10y", "base_rate"]:
            group_key_map[col] = "macro"
        elif "토픽" in col:
            group_key_map[col] = "topic"

    for col in ext_col_names + topic_col_names:
        col_sim = f"similar_{col}"
        if "date_close" in col:
            group_key_map[col_sim] = "price_close"
        elif "date_volume" in col:
            group_key_map[col_sim] = "volume"
        elif "date_foreign" in col:
            group_key_map[col_sim] = "foreign"
        elif "date_institution" in col:
            group_key_map[col_sim] = "institution"
        elif "date_individual" in col:
            group_key_map[col_sim] = "individual"
        elif col in ["fx", "bond10y", "base_rate"]:
            group_key_map[col_sim] = "macro"
        elif "토픽" in col:
            group_key_map[col_sim] = "topic"

    # 텍스트 임베딩 + AE 인코딩
    all_texts = [summary] + similar_summaries
    embeddings = np.array(await embedding_api_func(all_texts))

    embA, embBs = embeddings[0], embeddings[1:]
    latentA = run_ae(ae_sess, embA.reshape(1, -1))[0]
    latentBs = [run_ae(ae_sess, e.reshape(1, -1))[0] for e in embBs]

    # 스케일링
    extA_total = extA + topicA
    extA_col_names = ext_col_names + topic_col_names
    extA_scaled = scale_ext_grouped(
        extA_total, extA_col_names, "extA", scalers, group_key_map
    )

    extB_total = [ext + topic for ext, topic in zip(extBs, topicBs)]
    extB_col_names = [f"similar_{col}" for col in ext_col_names + topic_col_names]
    extBs_scaled = [
        scale_ext_grouped(extB, extB_col_names, "extB_similar", scalers, group_key_map)
        for extB in extB_total
    ]

    # 회귀 예측
    inputA_name = regressor_sess.get_inputs()[0].name
    inputB_name = regressor_sess.get_inputs()[1].name
    output_name = regressor_sess.get_outputs()[0].name

    scores = []
    for i, (latentB, extB_scaled) in enumerate(zip(latentBs, extBs_scaled)):
        if extB_scaled.shape[0] != 42:
            raise ValueError(
                f"extB_scaled 길이 이상함! 기대: 42, 실제: {extB_scaled.shape[0]} | index: {i}"
            )

        featA = np.concatenate([latentA, extA_scaled]).reshape(1, -1).astype(np.float32)
        featB = np.concatenate([latentB, extB_scaled]).reshape(1, -1).astype(np.float32)

        score = regressor_sess.run(
            [output_name], {inputA_name: featA, inputB_name: featB}
        )[0][0][0]
        scores.append(score)

    # 결과 반환
    results = list(zip(similar_summaries, scores, news_topk_ids))
    results.sort(key=lambda x: -x[1])  # score 기준 내림차순 정렬

    return [
        {
            "news_id": nid,
            "summary": summ,
            "wdate": "",
            "score": float(score),
            "rank": i + 1,
        }
        for i, (summ, score, nid) in enumerate(results)
    ]


def get_news_recommended(payload, request):
    """
    뉴스 후보군 추천
    """
    news_clicked_ids = payload.news_clicked_ids
    news_candidate_ids = payload.news_candidate_ids

    # 임베딩
    news_clicked_list = []
    news_candidate_list = []

    # 추론
    model_recommend = request.app.state.model_recommend
    outputs = model_recommend.run(
        None,
        {"clicked": news_clicked_list, "candidates": news_candidate_list},
    )

    scores = outputs[0]  # [1, 5]
    top_1 = scores[0].argsort()[::-1][:1][0]  # Top-1 추천 인덱스

    # 예측 결과 (Top-3 후보 인덱스):
    # [14  0  2 16 19 13 15  1 10  7  9  4 17  5  8  3  6 18 11 12]
    return top_1

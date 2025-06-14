import numpy as np
import re
from konlpy.tag import Okt
import numpy as np

from schemas.model import SimilarNewsItem


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


def get_news_embedding(text, request):
    """
    뉴스 본문을 임베딩하는 함수입니다.
    """
    tokenizer = request.app.state.tokenizer_embedding
    session = request.app.state.session_embedding

    # 입력 텍스트
    tokens = tokenizer.encode(text)
    input_ids = np.array([tokens.ids], dtype=np.int64)
    attention_mask = np.array([[1] * len(tokens.ids)], dtype=np.int64)

    # 추론
    embedding = session.run(
        ["sentence_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0]

    # (1, 768) → List[List[float]]
    return [[float(x) for x in embedding[0]]]


def get_news_similar_list(payload, request):
    """
    유사 뉴스 top_k
    """
    article = payload.article
    top_k = payload.top_k

    vectordb = request.app.state.vectordb

    # 검색
    results = vectordb.similarity_search_with_score(article, k=top_k)

    news_similar_list = []
    for doc, score in results:
        news_id = doc.metadata.get("news_id")
        wdate = doc.metadata.get("wdate")
        summary = doc.page_content

        if not news_id or not wdate or not summary:
            continue

        news_similar_list.append(
            SimilarNewsItem(
                news_id=news_id,
                summary=summary,
                wdate=wdate,
                score=round(1 - float(score), 2),
            )
        )

    return news_similar_list


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
        lda_topics[f"topic {index}"] = value

    return lda_topics

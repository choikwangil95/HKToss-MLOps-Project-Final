import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llama_cpp import Llama
from typing import List, Dict, Any

class CSVDummyVectorDB:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.texts = df['content'].tolist()
        self.embeddings = np.vstack(df['embedding'].apply(eval).to_numpy()).astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[str]:
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [self.texts[i] for i in indices[0]]

class NewsTossChatbot:
    def __init__(
        self,
        csv_path: str = "/app/db/news_v2_vector_202506122113.csv",
        embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        llm_model_path: str = "../models/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        top_k: int = 10,
        rerank_n: int = 5
    ):
        # 1. 임베딩 모델 초기화
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 2. 벡터DB 초기화
        self.vector_db = CSVDummyVectorDB(csv_path)
        
        # 3. Cross-Encoder 리랭커 초기화
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to("cpu")
        self.reranker_model.eval()
        
        # 4. LLaMA3 LLM 초기화
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        
        # 5. 파라미터 설정
        self.top_k = top_k
        self.rerank_n = rerank_n
        
        # 6. 프롬프트 템플릿
        self.prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        당신은 주식 투자자를 위한 뉴스 기반 정보 어시스턴트 챗봇 '뉴스토스'입니다.
        당신의 임무는 실시간 뉴스와 과거 유사사건 뉴스 데이터를 바탕으로,
        - 사용자의 투자 판단에 도움이 되는 정보를 제공하고,
        - 과거 유사사건, 해당 시기의 주가 흐름, 관련 리포트의 핵심 내용을 구체적으로 인용하며,
        - 미래 전망 질문에는 과거 사례를 근거로 신중하게 의견을 제시하는 것입니다.

        답변 작성 시 반드시 다음을 지켜주세요:
        1. 답변 내용 중 포함되는 과거 유사사건의 날짜, 사건명, 당시 주가 흐름(상승/하락/횡보 등), 주요 리포트 내용은 구체적으로 인용하세요.
        2. 미래 전망 질문에는 과거 유사사건을 근거로 논리적인 전망을 제시하세요.
        3. 답변 마지막에는 '⭐️투자 판단은 본인의 책임입니다.⭐️'라는 안내문을 추가하세요.
        4. 답변은 반드시 한글로, 명확하고 간결하게 작성하세요.
        5. 제공된 검색 결과(유사도 높은 과거 뉴스, 주가 데이터, 리포트 등)만 근거로 사용하세요. 근거가 없으면 '근거가 없는데 답변해도 될까? 이건 너의 소중한 돈이 걸린 문제야'라고 하세요.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        검색 결과: {context}
        질문: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def retrieve_docs(self, query: str) -> List[str]:
        """FAISS를 이용한 문서 검색"""
        query_emb = self.embedding_model.encode([query]).astype('float32')
        return self.vector_db.search(query_emb, self.top_k)

    def rerank_docs(self, query: str, docs: List[str]) -> List[str]:
        """Cross-Encoder를 이용한 문서 리랭킹"""
        pairs = [(query, doc) for doc in docs]
        inputs = self.reranker_tokenizer(
            [f"{q} [SEP] {d}" for q, d in pairs],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze().cpu().numpy()
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:self.rerank_n]]

    def generate_answer(self, context: str, question: str) -> str:
        """LLAMA3 직접 추론"""
        full_prompt = self.prompt_template.format(context=context, question=question)
        output = self.llm(full_prompt, max_tokens=512, stop=["<|eot_id|>"])
        return output['choices'][0]['text']

    def answer(self, query: str) -> Dict[str, str]:
        # 문서 검색 → 리랭킹 → LLM 응답 생성
        docs = self.retrieve_docs(query)
        reranked_docs = self.rerank_docs(query, docs)
        context = "\n\n".join(reranked_docs)
        answer = self.generate_answer(context, query)
        return {"answer": answer, "reranked_docs": reranked_docs}

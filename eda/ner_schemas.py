from pydantic import BaseModel
from typing import List

# 제목 + 본문을 받아서 종목명을 예측하는 요청
class StockPredictByFieldRequest(BaseModel):
    title: str
    content: str

# 종목명 예측 결과를 반환하는 응답
class StockPredictResponse(BaseModel):
    stocks: List[str]

from pydantic import BaseModel
from typing import List

# 뉴스 ID 기반 종목명 추출 응답
class StockMatchResponse(BaseModel):
    news_id: str           
    stocks: List[str]     

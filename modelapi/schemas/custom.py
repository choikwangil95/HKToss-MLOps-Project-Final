from fastapi import Query
from pydantic import BaseModel, field_validator, Field, conlist
from typing import Optional
from datetime import datetime, date
from typing import List, Union, Dict
from pydantic import validator
import ast

class NewsOut_v2_Metadata(BaseModel):
    news_id: str
    summary: str
    stock_list: Optional[List[Dict[str, str]]] = []
    stock_list_view: Optional[List[Dict[str, str]]] = []
    industry_list: Optional[List[Dict[str, str]]] = []
    impact_score: Optional[float]

    class Config:
        from_attributes = True

    @field_validator("stock_list", "stock_list_view", "industry_list", mode="before")
    @classmethod
    def parse_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                return [v]
        return v


class NewsOut_v2_External(BaseModel):
    news_id: str

    d_minus_5_date_close: Optional[float]
    d_minus_5_date_volume: Optional[float]
    d_minus_5_date_foreign: Optional[float]
    d_minus_5_date_institution: Optional[float]
    d_minus_5_date_individual: Optional[float]

    d_minus_4_date_close: Optional[float]
    d_minus_4_date_volume: Optional[float]
    d_minus_4_date_foreign: Optional[float]
    d_minus_4_date_institution: Optional[float]
    d_minus_4_date_individual: Optional[float]

    d_minus_3_date_close: Optional[float]
    d_minus_3_date_volume: Optional[float]
    d_minus_3_date_foreign: Optional[float]
    d_minus_3_date_institution: Optional[float]
    d_minus_3_date_individual: Optional[float]

    d_minus_2_date_close: Optional[float]
    d_minus_2_date_volume: Optional[float]
    d_minus_2_date_foreign: Optional[float]
    d_minus_2_date_institution: Optional[float]
    d_minus_2_date_individual: Optional[float]

    d_minus_1_date_close: Optional[float]
    d_minus_1_date_volume: Optional[float]
    d_minus_1_date_foreign: Optional[float]
    d_minus_1_date_institution: Optional[float]
    d_minus_1_date_individual: Optional[float]

    d_plus_1_date_close: Optional[float]
    d_plus_2_date_close: Optional[float]
    d_plus_3_date_close: Optional[float]
    d_plus_4_date_close: Optional[float]
    d_plus_5_date_close: Optional[float]

    fx: Optional[float]
    bond10y: Optional[float]
    base_rate: Optional[float]

    class Config:
        from_attributes = True  # ORM 지원
        
class SimpleImpactResponse(BaseModel):
    d_plus: List
    d_minus: List
    impact_score: float
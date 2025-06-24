from fastapi import Query
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from datetime import datetime, date
from typing import List, Union, Dict
from pydantic import validator
import ast


class User(BaseModel):
    user_id: str
    user_pnl: int
    asset: int
    invest_score: int
    member_stocks: Optional[List[Dict[str, str]]]

    class Config:
        orm_mode = True


class UserLog(BaseModel):
    id: int
    user_id: str
    news_id: str
    wdate: Optional[datetime]

    class Config:
        orm_mode = True

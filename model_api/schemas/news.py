from pydantic import BaseModel
from typing import List


class Embedding(BaseModel):
    embedding: List[float]

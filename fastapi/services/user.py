from sqlalchemy.orm import Session
from sqlalchemy import or_, desc, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import text
from models.user import UserLogModel, UserProfileModel
from fastapi.responses import JSONResponse
from schemas.news import News, NewsOut_v2_External, SimilarNewsV2
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import datetime
import json
import ast
from fastapi import HTTPException
import requests
from datetime import datetime, timedelta


def get_user_list(
    db: Session,
):
    user_list = db.query(UserProfileModel).all()

    return user_list


def get_user_detail(user_id: str, db: Session):
    user = (
        db.query(UserProfileModel).filter(UserProfileModel.user_id == user_id).first()
    )

    if user is None:
        return None

    return user


def get_user_log_list(user_id: str, db: Session):
    user_logs = db.query(UserLogModel).filter(UserLogModel.user_id == user_id).all()

    return user_logs

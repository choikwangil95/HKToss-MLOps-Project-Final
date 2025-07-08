from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from datetime import datetime
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from core.db import get_db
from typing import List
import json
from typing import Optional
import numpy as np
from starlette.concurrency import run_in_threadpool

from schemas.user import User, UserLog
from services.user import get_user_detail, get_user_list, get_user_log_list

router = APIRouter(
    prefix="/users", tags=["Users"], responses={404: {"description": "Not found"}}
)


@router.get(
    "/",
    response_model=list[User],
    summary="유저 목록 조회",
    description="유저 목록 조회",
)
async def get_user_list_router(
    db: Session = Depends(get_db),
):
    """
    유저 목록 조회
    """
    return await run_in_threadpool(
        get_user_list,
        db,
    )


@router.get(
    "/{user_id}",
    response_model=User,
    summary="유저 상세 조회",
    description="유저 상세 조회",
)
async def get_user_detail_router(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    유저 상세 조회
    """
    return await run_in_threadpool(
        get_user_detail,
        user_id,
        db,
    )


@router.get(
    "/{user_id}/logs",
    response_model=list[UserLog],
    summary="유저 뉴스 클릭 로그 목록 조회",
    description="유저 뉴스 클릭 로그 목록 조회",
)
async def get_log_list_router(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    유저 뉴스 클릭 로그 목록 조회
    """
    return await run_in_threadpool(
        get_user_log_list,
        user_id,
        db,
    )

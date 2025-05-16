# HK Toss MLOps 최종 프로젝트

🏠 실시간 주식 투자 AI 애널리스트

> 과거 유사 사건 뉴스 및 증권사 리포트 기반 주식 투자 판단 보조 시스템


### 팀원
- [🔗](https://github.com/Joo-an) 이주안
- [🔗](https://github.com/yoo754) 정혜진
- [🔗](https://github.com/choikwangil95) 최광일
- [🔗](https://github.com/yeaaaun) 한예은

### 목차
- [1 프로젝트 개요](#1-프로젝트-개요)
- [2 프로젝트 구조](#2-프로젝트-구조)
- [3 개발환경 구성](#3-개발환경-구성)
- [4 개발 프로세스](#4-개발-프로세스)

## 1 프로젝트 개요

### 문제정의
- T.B.D
  
## 2 프로젝트 구조
### 폴더구조
```markdown
📁 src
 ㄴ 📁 fastapi                  # FastAPI
 ㄴ 📁 ...                          # T.B.D
📄 requirements.txt
📄 docker-compose.yaml
```

### 아키텍쳐

![화면 캡처 2025-05-15 105248](https://github.com/user-attachments/assets/045c5829-80a4-423b-838d-55db84296bb6)


## 3 개발환경 구성
### 기술 스택
- **언어**: Python 3.11
- **패키지 관리**: Miniconda

### 프로젝트 설정

```bash
# 1 프로젝트 폴더 생성 및 저장소 초기화
mkdir <folder_name>
cd <folder_name>
git init

# 2 저장소 복제 및 동기화
git remote add origin https://github.com/choikwangil95/HKToss-MLOps-Proejct-Final.git
git pull origin main

# 3 가상환경 설정
conda create -n <env_name> python=3.11 pip
conda activate <env_name>

# 4 Jupyter Notebook 커널 설정
conda install ipykernel --update-deps --force-reinstall

# 5 requirements 설치
pip install -r requirements.txt
```

## 4 개발 프로세스

<img src="https://github.com/user-attachments/assets/ce06d476-6f07-4209-bf8e-3739d2801e9b" width="600px"/>

### 브랜치 관리
- `main` : 운영 환경
- `develop` : 개발 환경
- `feature` : 기능 개발

### 작업 흐름

```bash
# 1 최신 develop 브랜치 동기화
git checkout develop
git pull origin develop

# 2 새로운 기능 브랜치 생성
git checkout -b <feature_branch>

# 3 작업 후 변경 사항 저장
git add .
git commit -m "커밋 메시지"

# 4 develop 브랜치 병합 (충돌 확인 필수)
git checkout develop
git pull origin develop
git merge <feature_branch>

# 5 원격 저장소 반영
git push origin develop
```

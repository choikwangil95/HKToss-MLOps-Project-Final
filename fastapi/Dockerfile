FROM python:3.9

COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

WORKDIR /fastapi

COPY . /fastapi

EXPOSE 80

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]

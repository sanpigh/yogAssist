FROM python:3.8.6-buster

COPY api /api
COPY yogAssist /yogAssist
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY trained_models /trained_models

COPY configs /configs


COPY credentials.json /credentials.json


CMD uvicorn api.fast_kpts:app_kpts --host 0.0.0.0 --port $PORT

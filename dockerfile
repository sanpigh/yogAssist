FROM python:3.8.6-buster

RUN pip install -r requirements.txt

COPY api /api
COPY yogAssist /yogAssist
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip


COPY trained_models /trained_models

COPY configs /configs


COPY credentials.json /credentials.json


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

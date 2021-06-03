from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from yogAssist.utils import extract_keypoints_dictionnary_from_json_api
from yogAssist.scoring import Scoring, compute_asana_scoring
import cv2
from yogAssist.utils import decode_api_dictionnary

app_kpts = FastAPI()

app_kpts.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app_kpts.get("/")
def index(request: Request):
    return {"greeting": "Hello world"}


from fastapi import APIRouter, Request, Depends, File, UploadFile, Form


class Item(BaseModel):
    file: list
    name: str


@app_kpts.post("/keypoints_cosine")
def upload_image(kpts: Item):  # UploadFile = File(...)):
    print(kpts.file)
    print(kpts.name)
    path = f'./assets/{kpts.name}.txt'
    keypoints = extract_keypoints_dictionnary_from_json_api(path)
    scoring_api_1 = Scoring(keypoints, local=False)
    dict_api_1 = scoring_api_1.run()

    keypoints_2 = decode_api_dictionnary(kpts.file[0]['keypoints'])
    scoring_api_2 = Scoring(keypoints_2, local=False)
    dict_api_2 = scoring_api_2.run()



    output_json = compute_asana_scoring(scoring_api_1,scoring_api_2)

    return {"scores": output_json}

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

import cv2

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


    return {"filename": test}  #image.filename}

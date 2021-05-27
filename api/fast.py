from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index(request: Request):
    return {"greeting": "Hello world"}



from fastapi import APIRouter, Request, Depends, File, UploadFile, Form


class Item(BaseModel):
    title: str


class Image(BaseModel):
    file: str


@app.post("/keypoints_text")
def upload_image(test: Item):  # UploadFile = File(...)):

    return {"filename": test}  #image.filename}

import base64 as b64

@app.post("/keypoints_img")
async def create_file(
        #file: UploadFile = File(...)
        file: Image):

    img = b64.b64decode(file.file)
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(img.shape)
    return {
        "file_size": "all good"
    }

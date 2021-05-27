from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}



from fastapi import APIRouter, Request, Depends, File, UploadFile, Form


class Item(BaseModel):
    title: str


@app.post("/keypoints_text")
def upload_image(test: Item):  # UploadFile = File(...)):

    return {"filename": test}  #image.filename}


@app.post("/keypoints_img")
async def create_file(
                      file: str, # UploadFile = File(...),
                      token: str = Form(...)):
    return {
        "file_size": "all good"
    }

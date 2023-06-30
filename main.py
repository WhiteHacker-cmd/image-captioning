from typing import Union
from fastapi import FastAPI, UploadFile
from predict import (ImgCModel)



app = FastAPI()


IM = ImgCModel()

@app.get('/')
def home():
    return {"msg": "hello world!!!!"}



@app.post('/predict')
def predict(file: UploadFile):
    predicted = IM.predict(file.file)
    return {"imge_name": file.filename, "caption": predicted}
# FastAPI implementation
import os
from model.image_classifier import ImageClassifier
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import sys
import logging
from pydantic import BaseModel


class PredictionResponseDto(BaseModel):
    filename: str
    contentype: str
    likely_class: str


app = FastAPI()

model_dir = os.environ.get('MODEL_DIR', '/home/martin/Programming/Python/DeepLearning/Cifar10/models')
figure_dir = os.environ.get('MODEL_DIR', '/home/martin/Programming/Python/DeepLearning/Cifar10/figures')
name = 'test1'
image_classifier = ImageClassifier.load(filepath=f'{model_dir}/{name}.ckpt', name='cnn')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/", response_model=PredictionResponseDto)
async def predict(file: UploadFile = File(...)):

    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        predicted_class = image_classifier.predict(image)
        predicted_class = image_classifier.classes[predicted_class]

        logging.info(f"Predicted Class: {predicted_class}")
        return {
            "filename": file.filename,
            "contentype": file.content_type,
            "likely_class": predicted_class,
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

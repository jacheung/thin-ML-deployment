import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import List
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, ValidationError, validator
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.ml.model import Model, get_model
import matplotlib.image as mpimg

app = FastAPI()


# class PredictRequest(BaseModel):
#     data: int
#
#     # @validator("data")
#     # def check_dimensionality(cls, v):
#     #     for point in v:
#     #         if len(point) != n_features:
#     #             raise ValueError(f"Each data point must contain {n_features} features")
#     #
#     #     return v
#
#
class PredictResponse(BaseModel):
    data: int


@app.post("/predict",
          description="Predict MNIST image")
async def predict(file: UploadFile = File(...), model: Model = Depends(get_model)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format."
    await file.read()
    image = np.array(Image.open(file.file))
    y_pred = model.predict_single_image(image=image)
    print(y_pred)
    result = PredictResponse(data=y_pred.tolist())

    return result



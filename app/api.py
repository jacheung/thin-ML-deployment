import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, ValidationError, validator
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.ml.model import Model, get_model
import matplotlib.image as mpimg

router = APIRouter(prefix='/v1/thinMLdeployment')


class PredictRequest(BaseModel):
    data: int

    # @validator("data")
    # def check_dimensionality(cls, v):
    #     for point in v:
    #         if len(point) != n_features:
    #             raise ValueError(f"Each data point must contain {n_features} features")
    #
    #     return v


class PredictResponse(BaseModel):
    data: int


@router.post("/predict",
             description="Predict MNIST image")
def predict(image: PredictRequest, model: Model = Depends(get_model)):
    X = mpimg.imread(image)
    y_pred = model.predict_single_image(image=X)
    result = PredictResponse(data=y_pred.tolist())

    return result



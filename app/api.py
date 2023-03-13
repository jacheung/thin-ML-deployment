from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, ValidationError, validator
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from ml.model import Model
import matplotlib.image as mpimg


router = APIRouter(prefix='/v1/sandmass')


class PredictRequest(BaseModel):
    data: List[List[float]]

    @validator("data")
    def check_dimensionality(cls, v):
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")

        return v


class PredictResponse(BaseModel):
    data: List[float]


@app.post("/predict",
          description="Predict MNIST image")
def predict(image: PredictRequest, model: Model = Depends(get_model)):
    X = mpimg.imread(image)
    y_pred = model.predict_single_image(image=X)
    result = PredictResponse(data=y_pred.tolist())

    return result



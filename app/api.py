import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Depends
from app.ml.model import Model, get_model

route = FastAPI()

class PredictResponse(BaseModel):
    prediction: int


@route.post("/predict",
            description="Predict MNIST image")
async def predict(file: UploadFile = File(...), model: Model = Depends(get_model)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format."
    await file.read()
    image = np.array(Image.open(file.file))

    y_pred = model.predict_single_image(image=image)
    result = PredictResponse(prediction=y_pred.tolist())
    print(y_pred)

    return result

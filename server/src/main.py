import json
import io

import numpy as np
from keras.utils import img_to_array, load_img
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import pydantic
from fastapi import FastAPI, HTTPException, UploadFile, File


PATH_TO_MODEL = './model/best_model.h5'
PATH_TO_CLASS_INDICES = './model/class_indices.json'
TARGET_IMAGE_SIZE = (160, 160)


class ClassifyImageResponse(pydantic.BaseModel):
    class_: str = pydantic.Field(serialization_alias='class')


def load_class_indices(file_path):
    with open(file_path) as f:
        data = json.load(f)

    return {
        class_idx: class_name
        for class_name, class_idx in data.items()
    }


def load_and_preprocess_image(image_file, target_size):
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    it = datagen.flow(img_array, batch_size=1)
    batch = next(it)
    augmented_image = batch[0]
    return augmented_image


def predict(data):
    if len(data) != 1:
        raise HTTPException(
            status_code=500,
            detail=f'Количество изображений != 1: {len(data)}'
        )

    predictions = model.predict(data)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return class_idx_to_name[predicted_class]


async def classify_image(file: UploadFile = File(...)) -> ClassifyImageResponse:
    if file.content_type.startswith('image/'):
        img_data = await file.read()
        img = load_and_preprocess_image(io.BytesIO(img_data), TARGET_IMAGE_SIZE)

        result = predict(np.expand_dims(img, axis=0))
        return ClassifyImageResponse(class_=result)
    else:
        raise HTTPException(status_code=400, detail='Файл не является изображением')


app = FastAPI()
app.post('/classify-image')(classify_image)

model = load_model(PATH_TO_MODEL)
class_idx_to_name = load_class_indices(PATH_TO_CLASS_INDICES)
datagen = ImageDataGenerator(rescale=1./255)

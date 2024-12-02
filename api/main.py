from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tf_keras

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# origins= [
#     "http://localhost",
#     "http://localhost:3000",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


'''
prod_model = tf.keras.models.load_model("../models/my_model.keras")
beta_model = tf.keras.models.load_model("../models/my__model.keras")
'''


# MODEL = tf.keras.models.load_model("../models/1.keras")
MODEL = tf.keras.models.load_model("C:/potato_plant_disease_classification/models/1/my__model.keras")

# MODEL = tf.keras.layers.TFSMLayer("../models/my_model.keras", call_endpoint="serving_default")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)): #this will be a file , image of a potato plant leaf
    #bytes = await file.read() #1 servers 100 devices all sending images,(at a time using async await)
    image =read_file_as_image(await file.read()) #1 servers 100 devices all sending images,(at a time using async await)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {'class': predicted_class,
            'confidence': float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)
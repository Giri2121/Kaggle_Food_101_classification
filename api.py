from fastapi import FastAPI,File,UploadFile
from fastapi.templating import Jinja2Templates
import tensorflow as tf
#from fastapi.responses import Response
import os
#from PIL import Image
#from io import BytesIO
from Helper_functions import pred_and_plot
import uvicorn

#creating the instance
app = FastAPI()
templates = Jinja2Templates(directory="Templates")
'''
templates = Jinja2Templates(directory="Templates")

@app.get("/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
        <head>
            <title>Simple HTML app</title>
        </head>
        <body>
            <h1>Navigate to <a href="http://localhost:8000/form">/form</a></h1>
        </body>
    </html>
    """
    
    
    @app.get("/form")
def form_post(file: UploadFile = File(...)):
    return {"filename": file}
'''
img_store = []

def preprocess(image):
    base_path = r'C:\Users\vikassaigiridhar\Music\food_101_test\Test_Images'
    model = tf.keras.models.load_model('./Models')
    print('Model loaded')
    class_names = [name for name in os.listdir(base_path)]
    pred_class, pred_prob = pred_and_plot(model, image, class_names)
    return pred_class,pred_prob

@app.get("/img_test")
async def html_file_upload(file:UploadFile=File(...)):
    templates.TemplateResponse("index.html",context={"filename":file.filename,"fil":file.file})

@app.post("/predict_image")
async def store_image(file:UploadFile = File(...)):
    extension = file.filename.split('.')[-1] in ("jpg","jpeg","png")
    if not extension:
        return "Image must be jpg or png format!"

    content = await file.read()
    img_store.append(content)

    return {"filename":file.filename}


@app.get("/predict_image")
async def generate_pred():
    if len(img_store) > 0:
        pred_class,pred_prob = preprocess(img_store[-1])
        return f'predicted class is {pred_class} with probability {pred_prob}'
    return 'no image to predict'


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
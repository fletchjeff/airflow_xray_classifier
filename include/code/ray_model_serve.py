# File name: model_on_ray_serve.py
import re
import ray
from ray import serve
from transformers import pipeline
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
import base64




def predict_image(inputs):
    from PIL import Image
    import tensorflow as tf
    from io import BytesIO
    import numpy as np
    import base64
    model = tf.keras.models.load_model('data/models/20220118-130100/xray_classifier_model.h5')
    class_names = ['bacteria','normal','virus']
#im = Image.open("data/data/test/normal/IM-0117-0001.jpeg")
    im = Image.open(BytesIO(base64.b64decode(inputs)))
    im = im.resize((224,224),Image.ANTIALIAS)
    im = im.convert("RGB")
    im_array = tf.keras.preprocessing.image.img_to_array(im)
    im_array = np.expand_dims(im_array, axis=0)
    outputs = model.predict(im_array)
    output_classes = tf.math.argmax(outputs, axis=1)
    return {'result' : [class_names[c] for c in output_classes]}

args = { 
   "path" : "data/test/normal/IM-0117-0001.jpeg",
   "image" : "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABGgAAALwC"
}

def summarize(text):
    summarizer = pipeline("summarization", model="t5-small")
    summary_list = summarizer(text)
    summary = summary_list[0]["summary_text"]
    return summary


ray.init(address="auto", namespace="serve")
serve.start(detached=True,http_options={"host":"0.0.0.0"})


@serve.deployment
def router(request):
    txt = request.query_params["txt"]
    return summarize(txt)

predictor.delete()

@serve.deployment
async def predictor(request):
    from PIL import Image
    import tensorflow as tf
    from io import BytesIO
    import numpy as np
    import base64
    image_data = await request.json()
    print(image_data)
    model = tf.keras.models.load_model('data/models/20220118-130100/xray_classifier_model.h5')
    class_names = ['bacteria','normal','virus']
#im = Image.open("data/data/test/normal/IM-0117-0001.jpeg")
    im = Image.open(BytesIO(base64.b64decode(image_data['image'][22:])))
    im = im.resize((224,224),Image.ANTIALIAS)
    im = im.convert("RGB")
    im_array = tf.keras.preprocessing.image.img_to_array(im)
    im_array = np.expand_dims(im_array, axis=0)
    outputs = model.predict(im_array)
    output_classes = tf.math.argmax(outputs, axis=1)
    return {'result' : [class_names[c] for c in output_classes]}    
    # image = await request.json()
    # print(image['image'])
    # return "hello" #predict_image(image)

predictor.deploy()

from PIL import Image
from io import BytesIO
import numpy as np
import base64
import json, requests, base64
#data = 'data:image/png;base64,' + image_data.decode()
#encoded = base64.b64encode(open("data/data/test/normal/IM-0117-0001.jpeg", "rb").read()).decode('ascii')
im = Image.open("data/data/test/normal/IM-0117-0001.jpeg")
output = BytesIO()
im.save(output, format='PNG')
im_data = output.getvalue()
image_data = base64.b64encode(im_data)
#data = 'data:image/png;base64,' + image_data.decode()
#data = json.dumps({'request' : {'image':data}})
data = json.dumps({
   "path" : "data/test/normal/IM-0117-0001.jpeg",
   "image" : f"data:image/png;base64,{image_data.decode()}"
})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://a48892de9f3b44901a5fbf0d47d2f24a-100577308.eu-central-1.elb.amazonaws.com:8000/predictor', data=data, headers=headers)
result = json_response.json()







TFMnistModel.delete()

@serve.deployment(route_prefix="/predict_2")
class TFMnistModel:
    def __init__(self, model_path):
        #import tensorflow as tf
        self.model_path = model_path
        #self.model = tf.keras.models.load_model(model_path)
    async def __call__(self, starlette_request):
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        #input_array = (await starlette_request.json())["array"]
        input_array = await starlette_request.body() #.json()# np.array((await starlette_request.json())["array"])
        # reshaped_array = input_array.reshape((1, 28, 28))
        # # Step 2: tensorflow input -> tensorflow output
        # prediction = self.model(reshaped_array)
        # Step 3: tensorflow output -> web output
        return input_array #{"prediction": prediction.numpy().tolist(), "file": self.model_path}

TFMnistModel.deploy("data/models/20220118-130100/xray_classifier_model.h5")


predictor.deploy()

router.deploy()


#encoded = base64.b64encode(open("data/data/test/normal/IM-0117-0001.jpeg", "rb").read())
encoded = base64.b64encode(open("data/data/test/normal/IM-0117-0001.jpeg", "rb").read()).decode('ascii')
args = { "request" : {
   "path" : "data/test/normal/IM-0117-0001.jpeg",
   "image" : f"data:image/png;base64,{encoded}"
}
}

import requests

response = requests.get("http://192.168.1.100:8000/predictor?image=" + encoded).text

print(response)

# from io import BytesIO  
# output = BytesIO()
# im.save(output, format='JPEG')
# im_data = output.getvalue()
# #This you can then encode to base64:

# image_data = base64.b64encode(im_data)
# if not isinstance(image_data, str):
#     # Python 3, decode from bytes to string
#     image_data = image_data.decode()
# data_url = 'data:image/jpg;base64,' + image_data

import bentoml
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import json

from bentoml.artifact import KerasModelArtifact
from bentoml.adapters import JsonInput,JsonOutput
from bentoml.types import JsonSerializable, InferenceTask, InferenceResult, InferenceError 

class_names = ['bacteria','normal','virus']

@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow'])
@bentoml.artifacts([KerasModelArtifact('model')])
class TensorflowXray(bentoml.BentoService):
    @bentoml.api(input=JsonInput(),batch=False,output=JsonOutput())#input=TfTensorInput(), batch=True)
    def predict(self, inputs):
        print(inputs)
        im = Image.open(BytesIO(base64.b64decode(inputs['request']['image'][22:])))
        im = im.resize((224,224),Image.ANTIALIAS)
        im = im.convert("RGB")
        im_array = tf.keras.preprocessing.image.img_to_array(im)
        im_array = np.expand_dims(im_array, axis=0)
        outputs = self.artifacts.model.predict(im_array)
        output_classes = tf.math.argmax(outputs, axis=1)
        return {'result' : [class_names[c] for c in output_classes]}

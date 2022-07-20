import ray
from ray import serve

ray.init("ray://a48892de9f3b44901a5fbf0d47d2f24a-100577308.eu-central-1.elb.amazonaws.com:10001")
serve.start(detached=True,http_options={"host":"0.0.0.0"})

@serve.deployment
async def predictor(request):
    from PIL import Image
    import tensorflow as tf
    from io import BytesIO
    import numpy as np
    import base64
    image_data = await request.json()
    model = tf.keras.models.load_model('/data/models/20220719-125500/xray_classifier_model.h5')
    class_names = ['bacteria','normal','virus']
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

if 'predictor' in serve.list_deployments().keys():
    predictor.delete()

predictor.deploy()

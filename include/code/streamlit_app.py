import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import random
import base64
from io import BytesIO
import json
import requests
from lime import lime_image
from skimage.segmentation import mark_boundaries


# Do not delete these lines, they're need udpated by Airflow
RAY_SERVER=''
STORAGE_PATH=''
CURRENT_RUN=''
st.header("MLOps with Aiflow: Xray Classifier")
st.text(f"Model = {CURRENT_RUN}")


def random_image():
  normal_file = glob.glob(f"{STORAGE_PATH}/data/test/normal/*.jpeg")[:200]
  pneumonia_file = glob.glob(f"{STORAGE_PATH}/data/test/pneumonia/*.jpeg")[:200]
  all_files = normal_file + pneumonia_file
  return random.choice(all_files)

def get_new_image():
    st.session_state.image = random_image()
    st.session_state.prediction_value = "--"
    st.session_state.actual_value = st.session_state.image.split("/test")[1].split("/")[1]

def predict_image():
  if type(st.session_state.image) == str:
    im = Image.open(st.session_state.image)
    output = BytesIO()
    im.save(output, format='PNG')
    im_data = output.getvalue()
    image_data = base64.b64encode(im_data)
    data = 'data:image/png;base64,' + image_data.decode()
    data = json.dumps({'image':data,'current_run':CURRENT_RUN})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f'http://{RAY_SERVER}:8000/predictor', data=data, headers=headers)
    result = json_response.json()
    st.session_state.prediction_value = result["result"]
  else:
    st.session_state.prediction_value = "Can't predict an explained image"

def explain_image():
  model = tf.keras.models.load_model(f'{STORAGE_PATH}/models/{CURRENT_RUN}/xray_classifier_model.h5')
  explainer = lime_image.LimeImageExplainer()
  xray_image = Image.open(st.session_state.image)
  original_size = xray_image.size
  xray_image = xray_image.resize((224,224),Image.ANTIALIAS)
  xray_image = xray_image.convert("RGB")
  xray_image_array = tf.keras.preprocessing.image.img_to_array(xray_image)
  xray_image_array = np.expand_dims(xray_image_array, axis=0)
  explanation = explainer.explain_instance(xray_image_array[0].astype('double'), model.predict)
  _, explain_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)

  blank_image = Image.new('RGB', (224, 224))
  blank_image_array = tf.keras.preprocessing.image.img_to_array(blank_image)
  explained_image = mark_boundaries(blank_image_array,explain_mask,mode='thick',color=(0,0.75,1))
  explained_image = Image.fromarray((explained_image * 255).astype(np.uint8))
  explained_image = explained_image.convert("RGBA")

  # Create Alpha Channel
  explained_image_data = explained_image.getdata()
  newData = []
  for item in explained_image_data:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((0, 0, 0, 0))
    else:
        newData.append(item)
  explained_image.putdata(newData)
  explained_image = explained_image.resize(original_size,Image.ANTIALIAS)
  original_xray_image = Image.open(st.session_state.image)
  original_xray_image = original_xray_image.convert("RGB")
  original_xray_image.paste(explained_image,(0,0),explained_image)
  st.session_state.image = original_xray_image

if 'image' not in st.session_state:
    st.session_state.image = random_image()
    st.session_state.prediction_value = "--"
    st.session_state.actual_value = st.session_state.image.split("/test")[1].split("/")[1]

col1, col2, col3 = st.columns([1,1,3])
col1.button('Get New Image',on_click=get_new_image)
col2.button('Predict via API', on_click=predict_image)
col3.button('Explain Image', on_click=explain_image)
actual = st.write("Actual Value: {}".format(st.session_state.actual_value))
prediction = st.write("Predict Value: {}".format(st.session_state.prediction_value))
container = st.container()
main_image = container.image(st.session_state.image,clamp=True)
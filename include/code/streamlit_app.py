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

st.title("Xray Classifier")
host_ip = "192.168.1.100"

def random_image():
  # normal_file = glob.glob("/mnt/data/xray/data/test/normal/*.jpeg")
  # bacteria_file = glob.glob("/mnt/data/xray/data/test/bacteria/*.jpeg")
  # virus_file = glob.glob("/mnt/data/xray/data/test/virus/*.jpeg")
  normal_file = glob.glob("data/data/test/normal/*.jpeg")
  bacteria_file = glob.glob("data/data/test/bacteria/*.jpeg")
  virus_file = glob.glob("data/data/test/virus/*.jpeg")  
  all_files = normal_file + bacteria_file + virus_file
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
    data = json.dumps({'image':data})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f'http://{host_ip}:8000/predictor', data=data, headers=headers)
    result = json_response.json()
    st.session_state.prediction_value = result["result"][0]
  else:
    st.session_state.prediction_value = "Can't predict an explained image"

def explain_image():
  model = tf.keras.models.load_model('include/models/xray_classifier_model.h5')
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
main_image = container.image(st.session_state.image,clamp=True)#.image








# def teachable_machine_classification(img, weights_file):
#     # Load the model
#     model = tf.keras.models.load_model(weights_file)

#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
#     image = img
#     #image sizing
#     size = (200, 200)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)

#     #turn the image into a numpy array
#     image_array = np.asarray(image)
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 255)

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     prediction_percentage = model.predict(data)
#     prediction=prediction_percentage.round()
    
#     return  prediction,prediction_percentage


# uploaded_file = st.file_uploader("Choose an Cat or Dog Image...", type="jpg")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded file', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#     label,perc = teachable_machine_classification(image, 'catdog.h5')
#     if label == 1:
#         st.write("Its a Dog, confidence level:",perc)
#     else:
#         st.write("Its a Cat, confidence level:",1-perc)
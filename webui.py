# Web App UI using streamlit
import streamlit as st 
import pandas as pd
import numpy as np
import json
from fastai.vision.all import *
from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# global Tf
# Tf = None

def get_x(row): return row['image_id']
def get_y(row): return row['label']

def classify_image(img):
    mapping = {0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevus',
    5: 'Melanoma',
    6: 'Squamous cell carcinoma',
    7: 'Vascular lesion',
    8: 'Normal'}
    print(img)
    pred_idx,_,probs = learn.predict(img)
    pred_class = mapping[int(pred_idx)]
    prob = probs.max()
    prob = prob.numpy()
    return pred_class,prob


st.set_page_config(page_title="EpiCheck", page_icon=":guardsman:", layout="wide")

st.title("EpiCheck ~ Automated Skin Disease Detection System")
st.text("A web platform to detect skin diseases using Deep Learning")

# load your pre-trained model
option = st.selectbox(
'Selected Model',
('Resnet152', 'VGG16'))
if option == 'Resnet152':
    learn = load_learner('own_resnet.pkl')
elif option == 'VGG16':
    learn = load_learner('own_vgg.pkl')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    print(uploaded_file.name)
    print(image)
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    st.image(image.resize((400,400)), caption='Uploaded Image.', use_column_width=False)
    # st.write(file_details)
    saved_file_path = os.path.join("tempDir",uploaded_file.name) 
    with open(saved_file_path,"wb") as f: 
        f.write(uploaded_file.getbuffer())         
    # st.success("Saved File")
    st.write("Classifying...")
    pred_class,prob = classify_image(saved_file_path)
    st.success(f"The model predicts the image as: {pred_class} with probability: {prob*100:.2f}%")
    description_file = open('comment.json')
    disease_details = json.load(description_file)
    cancers = ['Basal cell carcinoma', 'Melanoma', 'Squamous cell carcinoma']
    if pred_class in cancers:
        st.write("### Skin Cancer Detected. Please consult a doctor immediately.")
    elif pred_class=="Normal":
        st.write("### No Skin disease found. Please consult a doctor if you have any concerns.")
    else:
        st.write("### This is a Non-Cancerous Skin disease. Please consult a doctor for further treatment.")
    st.write("Please Note that Our system is still under development so the results may not be accurate. Please consult a doctor for verification and treatment.")
    # get descriptions
    if pred_class != "Normal":
        description = disease_details[pred_class]["Description"]
        symptoms = disease_details[pred_class]["Symptoms"]
        causes = disease_details[pred_class]["Causes"]
        risk_factors = disease_details[pred_class]["Risk Factors"]
        treatment = disease_details[pred_class]["Treatment"]
        diagnosis = disease_details[pred_class]["Diagnosis"]
        treatment = disease_details[pred_class]["Treatment"]
        st.write("## Disease Description ")
        st.write(description)
        st.write("## Symptoms ")
        st.write(symptoms)
        st.write("## Possible Causes ")
        st.write(description)
        st.write("## Risk Factors ")
        st.write(risk_factors)
        st.write("## Diagnosis ")
        st.write(diagnosis)
        st.write("## Treatment ")
        st.write(treatment)
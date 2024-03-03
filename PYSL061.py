import streamlit as st
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model('Pokemon.h5',compile=False)

target_img_shape=(128,128)

st.subheader("การจำแนกภาพดอกไม้")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #------------------------------------------------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
    test_image = cv2.resize(img,target_img_shape)
    
    test_image = img_to_array(test_image)
    test_image = preprocess_input(test_image)

    test_image = np.expand_dims(test_image,axis=0) # (1, 128, 128, 3)

    result = model.predict(test_image)
    st.write(result)

    class_answer = np.argmax(result,axis=1)
    if class_answer == 0:
        predict = 'rose'
    elif class_answer == 1:
        predict = 'sunflower'
    elif class_answer == 2:
        predict = 'tulip'
    elif class_answer == 3:
        predict = 'Pikachu'
    elif class_answer == 4:
        predict = 'Squirtle'
       
    st.write("predict = "+predict)
    #------------------------------------------------
    st.image(img ,caption=predict,channels="RGB")

    

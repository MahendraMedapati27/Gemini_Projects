from dotenv import load_dotenv
load_dotenv() ## load all the environment variables from .env

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro-vision")

def get_response(prompt, image, input):
    response = model.generate_content([prompt,image[0], input])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
st.set_page_config(page_title="Gemini Vision Doctor")

st.header("Gemini Vision Doctor")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image of the body part that is effected or you are suffering from...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit=st.button("Tell me about the disease")
input_prompt = """"
You're proficient in diagnosing diseases based on images of the patient's body parts. 
Upon analyzing the images, you determine the disease the patient is suffering from and provide
recommendations accordingly. You prescribe medications and suggest dietary changes tailored to treating
the identified disease. In severe cases, you inform the patient about the necessity of undergoing surgery for
treatment.
"""
if submit:
    image_data=input_image_details(uploaded_file)
    response=get_response(input_prompt,image_data, input)
    st.subheader("The Response is")
    st.write(response)
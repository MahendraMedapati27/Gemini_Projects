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
    

st.set_page_config(page_title="Gemini Nutritionist")

st.header("Gemini Nutritionist")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image of the Food...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit=st.button("Tell me about the total calories")

input_prompt="""
You are an expert in nutrionist where you need to see the food items fro the image
and calculate the total calories also provide the details of evry food item with calories intake
in below format
1. Item 1 - no of calories
2. Item 2 - no of calories
----
-----

finally you can also mention wheater the food is healthy or not and also mention the percentage split of the ratio of carbohydrates, fats, fibres, sugars and 
other things required in our diet and you should recommand better diet than the image
"""

## if submit button is clicked

if submit:
    image_data=input_image_details(uploaded_file)
    response=get_response(input_prompt,image_data, input)
    st.subheader("The Response is")
    st.write(response)
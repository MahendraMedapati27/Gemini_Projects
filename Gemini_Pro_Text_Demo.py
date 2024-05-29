from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
def get_response_from_gemini(question):
    response = model.generate_content(question)
    return response.text

st.set_page_config(page_title="Q&A demo")
st.header("Gemini Pro Text aplication")
input = st.text_input("Input: ", key="Input")
submit = st.button("Ask the question")

if submit:
    response = get_response_from_gemini(input)
    st.subheader("Your response is ")
    st.write(response)
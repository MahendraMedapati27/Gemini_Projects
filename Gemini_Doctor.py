import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the Google API key for generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the generative model
model = genai.GenerativeModel("gemini-pro")

# Define the function to get the AI response
def get_response(input_text, prompt):
    response = model.generate_content([input_text, prompt])
    return response

# Set the page configuration
st.set_page_config(page_title="Gemini Doctor")

# App header
st.header("Gemini Doctor")

# User input for symptoms or questions
user_input = st.text_input("Describe your symptoms or ask a health-related question:", key="input")

# Submit button
submit = st.button("Ask")

# Input prompt for the AI
input_prompt = """
As a renowned doctor with comprehensive knowledge of various diseases and their causes, you have an 
exceptional ability to diagnose illnesses based on patient symptoms. In addition to diagnosing, you
recommend appropriate medicines and dietary changes to treat the disease. For severe conditions, you 
also advise on the necessity of surgical intervention if required.
"""

# Initialize session state to store query history
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Function to handle the submission and response display
def handle_submission():
    if user_input:
        response = get_response(user_input, input_prompt)
        if hasattr(response, 'text'):
            st.session_state.query_history.append((user_input, response.text))
            st.success("Response received! Scroll down to see it.")
        else:
            st.warning("The AI response may have been blocked or contains sensitive content.")
    else:
        st.error("Please enter your symptoms or question.")

# Handle the submission
if submit:
    handle_submission()

# Display the history of queries and responses
if st.sidebar.checkbox("View Search History", False):
    st.sidebar.subheader("Search History")
    for i, (query, response) in enumerate(st.session_state.query_history, 1):
        st.sidebar.markdown(f"**Query {i}:** {query}")
        st.sidebar.markdown(f"**Response {i}:** {response}")
        st.sidebar.write("---")

# Display only the current query and response
if st.session_state.query_history:
    current_query, current_response = st.session_state.query_history[-1]
    st.subheader("Current Query and Response")
    st.markdown(f"**Query:** {current_query}")
    st.markdown(f"**Response:** {current_response}")

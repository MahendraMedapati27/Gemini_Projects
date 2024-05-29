import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the generative model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get a response from the model
def get_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Streamlit page configuration
st.set_page_config(page_title="Q&A Demo", layout="wide")
st.title("Gemini LLM Application")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Layout for input and button
input_col, button_col = st.columns([4, 1])
with input_col:
    input_text = st.text_input("Input your question here:", key="input")
with button_col:
    submit = st.button("Ask me")

# Handle user input and generate response
if submit and input_text:
    response = get_response(input_text)
    st.session_state['chat_history'].append(("You", input_text))
    for chunk in response:
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display the latest response
if st.session_state['chat_history']:
    st.subheader("The Response is")
    last_role, last_text = st.session_state['chat_history'][-1]
    if last_role == "Bot":
        st.write(last_text)

# Display chat history using chat messages
st.subheader("The chat history is")
for role, text in st.session_state['chat_history']:
    if role == "You":
        st.markdown(f"<div style='text-align: right;'><b>{role}:</b> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'><i>{role}:</i> {text}</div>", unsafe_allow_html=True)

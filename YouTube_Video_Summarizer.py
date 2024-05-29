import streamlit as st
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI library
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt_text = """You are a YouTube video summarizer. You will be given the transcript text and should summarize the entire video, providing the important points within 300 words. The transcript text will be appended here: """

def extract_video_id(youtube_video_url):
    """Extracts the video ID from a YouTube URL."""
    if "youtube.com/watch?v=" in youtube_video_url:
        return youtube_video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_video_url:
        return youtube_video_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")

def extract_transcript(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all the transcript parts into a single string
        transcript = " ".join([item["text"] for item in transcript_list])
        
        return transcript
    except Exception as e:
        raise e

def get_response(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Set the page configuration
st.set_page_config(page_title="English YouTube Video Summarizer")

# Header for the Streamlit app
st.header("YouTube Video to Summary Converter")

# Input field for the YouTube video link
video_link = st.text_input("Enter your YouTube video link here:")

# Display the video if a link is provided
if video_link:
    st.video(video_link)

# Button to get the summary of the video
button = st.button("Get the Summary of the Video")

if button and video_link:
    try:
        # Extract transcript
        transcript = extract_transcript(video_link)
        # Get response from the generative AI model
        response = get_response(transcript, prompt_text)
        # Display the summary
        st.subheader("Summary of the provided Video is:")
        st.write(response)
    except ValueError as ve:
        st.error(f"An error occurred: {ve}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

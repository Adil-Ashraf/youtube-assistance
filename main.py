import streamlit as st
import langchain_helper as lch
from utils.url_utils import is_valid_youtube_url


st.title("DevBox Video Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="Enter the YouTube video URL here!",
            max_chars=100
            )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=100,
            key="query"
            )
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if not youtube_url or not is_valid_youtube_url(youtube_url):
                st.error("Please enter a valid YouTube URL.")
            if not query:
                st.error("Please enter a question.")

if submit_button and youtube_url and is_valid_youtube_url(youtube_url) and query:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    st.subheader("Answer:")
    chat_box = st.empty()
    lch.get_response_from_query(db, query, chat_box)

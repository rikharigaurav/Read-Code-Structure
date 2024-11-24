import streamlit as st
import requests
import json

st.title("Github Structure Reader")
githubLink = st.text_input("GitHub Repo Link")
docsURL = st.text_input("Docs url")

inputs = {
    "repoUrl": githubLink,
    "docsUrl": docsURL
}

if st.button("Select repo"):
    res = requests.post(url = f"http://127.0.0.1:3000/github/", data = json.dumps(inputs))
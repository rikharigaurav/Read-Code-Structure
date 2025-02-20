import streamlit as st
import requests
import json

st.title("Github Structure Reader")

# Initialize session state
if 'repo_selected' not in st.session_state:
    st.session_state.repo_selected = False
    st.session_state.issues = []

# Input form
if not st.session_state.repo_selected:
    githubLink = st.text_input("GitHub Repo Link")
    docsURL = st.text_input("Docs url")

    inputs = {
        "repoUrl": githubLink,
        "docsUrl": docsURL
    }

    if st.button("Select repo"):
        # Send request to backend
        res = requests.post(url="http://127.0.0.1:3000/github/", data=json.dumps(inputs))
        
        if res.status_code == 200:
            # Store issues in session state (assuming response contains issues)
            try:
                st.session_state.issues = res.json().get('issues', [])
                st.session_state.repo_selected = True
            except json.JSONDecodeError:
                st.error("Invalid response from server")
        else:
            st.error(f"Error fetching repository data: {res.status_code}")

# Display issues after repo selection
else:
    st.subheader("Open GitHub Issues")
    
    if st.button("Back to Repository Selection"):
        st.session_state.repo_selected = False
        st.session_state.issues = []
        st.experimental_rerun()
    
    if not st.session_state.issues:
        st.write("No open issues found!")
    else:
        for issue in st.session_state.issues:
            # Customize based on your API's response structure
            st.markdown(f"### {issue.get('title', 'Untitled Issue')}")
            st.write(f"**State:** {issue.get('state', 'N/A')}")
            st.write(f"**Created At:** {issue.get('created_at', 'N/A')}")
            st.write(f"**URL:** [{issue.get('html_url', '#')}]({issue.get('html_url', '#')})")
            st.write(f"**Description:** {issue.get('body', 'No description')[:200]}...")
            st.markdown("---")
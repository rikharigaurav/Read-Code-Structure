import streamlit as st
import requests
import os
import git
from github import Github
import json

# Configure Streamlit page
st.set_page_config(page_title="GitHub Code Parser", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'repo_data' not in st.session_state:
    st.session_state.repo_data = None
if 'selected_issue' not in st.session_state:
    st.session_state.selected_issue = None

def clone_repo(repo_url):
    """Clone the repository and return the path"""
    try:
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        local_path = f"./cloned_repos/{repo_name}"
        
        # Clone the repository
        if not os.path.exists(local_path):
            git.Repo.clone_from(repo_url, local_path)
        
        return local_path
    except Exception as e:
        st.error(f"Error cloning repository: {str(e)}")
        return None

def get_repo_issues(repo_url):
    """Get issues from GitHub repository"""
    try:
        # Extract owner and repo name from URL
        parts = repo_url.split('/')
        owner = parts[-2]
        repo = parts[-1].replace('.git', '')
        
        # Use GitHub API to get issues
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        response = requests.get(api_url)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching issues: {str(e)}")
        return []

def handle_submit():
    """Handle form submission"""
    repo_url = st.session_state.repo_url
    
    # Make API POST request (replace with your actual API endpoint)
    api_endpoint = "http://127.0.0.1:3000/test/"
    try:
        response = requests.post(api_endpoint, json={"repo_url": repo_url})
        if response.status_code == 200:
            # Clone repo and get issues
            # local_path = clone_repo(repo_url)
            local_path = 'Read-Code-Structure'
            issues = get_repo_issues(repo_url)
            
            # Store data in session state
            st.session_state.repo_data = {
                "local_path": local_path,
                "issues": issues
            }
            
            # Change page
            st.session_state.page = 'repo_view'
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")

def build_tree(root_dir: str) -> dict:
    tree = {}
    for root, dirs, files in os.walk(root_dir):
        # Get path relative to the root_dir
        relative_path = os.path.relpath(root, root_dir)
        if relative_path == ".":
            relative_path = ""
        
        # Navigate into the nested dictionary based on subfolder path
        path_parts = relative_path.split(os.sep) if relative_path else []
        current_dict = tree
        for part in path_parts:
            current_dict = current_dict.setdefault(part, {})
        
        # Add files into the current_dict
        for file in files:
            current_dict[file] = None  # Files map to None (or you could store file metadata)
    
    return tree


def display_tree(tree: dict, path: str = "", level: int = 0):
    """
    Displays one level of expanders (for top-level folders),
    and for deeper levels, just shows an indented list of folders/files.
    """
    indent = " " * (level * 3)  # adjust indentation as needed

    for key, value in tree.items():
        if isinstance(value, dict):
            # If this is the top level, use an expander
            if level == 0:
                with st.expander(f"📂 {key}", expanded=False):
                    display_tree(value, os.path.join(path, key), level + 1)
            else:
                # Just show an indented folder name (no nested expander)
                st.write(f"{indent}📂 {key}")
                display_tree(value, os.path.join(path, key), level + 1)
        else:
            # It's a file, just show an indented file name
            st.write(f"{indent}📄 {key}")



def show_home_page():
    """Render home page"""
    st.title("GITHUB CODE PARSER")
    
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("PASTE YOUR GITHUB REPO LINK", key="repo_url")
        st.button("SUBMIT", on_click=handle_submit)

def show_repo_page():
    """Render repository view page"""
    if not st.session_state.repo_data:
        st.session_state.page = 'home'
        return
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Tabs for Folder Content and Issues
        tab1, tab2 = st.tabs(["FOLDER CONTENT", "ISSUES"])
        
        with tab1:
            local_path = st.session_state.repo_data["local_path"]
            
            if local_path:
                # Build the nested dictionary
                tree = build_tree(local_path)
                
                # Recursively display it
                display_tree(tree)
            else:
                st.write("No local path found.")
        
        with tab2:
            for issue in st.session_state.repo_data["issues"]:
                if st.button(f"#{issue['number']} - {issue['title']}", key=issue['number']):
                    st.session_state.selected_issue = issue
    
    with col2:
        st.header("CHAT BOX TO CHAT WITH LLM")
        if st.session_state.selected_issue:
            st.write(f"Selected Issue: #{st.session_state.selected_issue['number']}")
            # Add your chat interface here
            user_input = st.text_input("Your message")
            if st.button("Send"):
                # Add your LLM integration here
                pass

# Main app logic
def main():
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'repo_view':
        show_repo_page()

if __name__ == "__main__":
    main()
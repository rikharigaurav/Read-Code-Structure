import streamlit as st
import requests
import os
import git
from github import Github
import json
from datetime import datetime

# Configure Streamlit page with dark theme
st.set_page_config(page_title="GitHub Code Parser", layout="wide")

# Apply custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #f0f0f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #990000;
        color: white;
    }
    .folder-tree {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 4px;
    }
    .code-viewer {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 10px;
    }
    .chat-bot {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 10px;
    }
    .file-item:hover {
        background-color: #333333;
        cursor: pointer;
    }
    .selected-file {
        background-color: #990000;
        border-radius: 4px;
        padding: 2px 5px;
    }
    .issue-card {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .issue-number {
        color: #990000;
        font-weight: bold;
    }
    .search-container {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .issue-btn {
        text-align: left;
        background-color: #1e1e1e;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 5px;
        cursor: pointer;
        width: 100%;
    }
    .issue-btn:hover {
        background-color: #333333;
    }
    .search-btn {
        background-color: #990000;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .search-input {
        background-color: #2c2c2c;
        color: white;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'repo_data' not in st.session_state:
    st.session_state.repo_data = None
if 'selected_issue' not in st.session_state:
    st.session_state.selected_issue = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'issue_search' not in st.session_state:
    st.session_state.issue_search = ""
if 'filtered_issues' not in st.session_state:
    st.session_state.filtered_issues = []

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
    
    # Make API POST request to your backend
    try:
        response = requests.post("http://127.0.0.1:3000/github/", json={"repo_url": repo_url})
        if response.status_code == 200:
            response_data = response.json()
            local_path = response_data.get("localPath")
            
            # Get GitHub issues
            issues = get_repo_issues(repo_url)
            
            # Store data in session state
            st.session_state.repo_data = {
                "local_path": local_path,
                "issues": issues
            }
            
            # Initialize filtered issues with all issues
            st.session_state.filtered_issues = issues
            
            st.session_state.page = 'repo_view'
        else:
            st.error(f"API request failed with status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")

def build_tree(root_dir: str) -> dict:
    """Build a nested dictionary representing the file structure"""
    tree = {}
    for root, dirs, files in os.walk(root_dir):
        # Get path relative to the root_dir
        relative_path = os.path.relpath(root, root_dir)
        if relative_path == ".":
            relative_path = ""
        
        # Skip .git directory and other hidden folders
        if '.git' in relative_path.split(os.sep):
            continue
        
        # Navigate into the nested dictionary based on subfolder path
        path_parts = relative_path.split(os.sep) if relative_path else []
        current_dict = tree
        for part in path_parts:
            if part.startswith('.'):
                continue
            current_dict = current_dict.setdefault(part, {})
        
        # Add files into the current_dict (skip hidden files)
        for file in files:
            if not file.startswith('.'):
                current_dict[file] = None
    
    return tree

def read_file_contents(file_path):
    """Read and return the contents of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def add_chat_message(text, is_user=True):
    """Add a message to the chat history"""
    st.session_state.chat_messages.append({
        'text': text,
        'is_user': is_user,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

def search_issues():
    """Filter issues based on search query"""
    search_query = st.session_state.issue_search.lower()
    if search_query:
        st.session_state.filtered_issues = [
            issue for issue in st.session_state.repo_data["issues"]
            if (search_query in issue['title'].lower() or 
                (issue['body'] and search_query in issue['body'].lower()) or
                str(issue['number']) == search_query)
        ]
    else:
        st.session_state.filtered_issues = st.session_state.repo_data["issues"]

def select_issue(issue):
    """Set the selected issue"""
    st.session_state.selected_issue = issue

def show_home_page():
    """Render home page"""
    st.title("GITHUB CODE PARSER")
    
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("PASTE YOUR GITHUB REPO LINK", key="repo_url", 
                     placeholder="e.g., https://github.com/username/repo")
        st.button("SUBMIT", on_click=handle_submit)

def display_tree_with_file_selection(tree: dict, path: str = "", local_path: str = ""):
    """Display folder tree with clickable files"""
    for key, value in sorted(tree.items(), key=lambda x: (isinstance(x[1], dict), x[0])):
        if isinstance(value, dict):
            # It's a directory
            with st.expander(f"📂 {key}", expanded=False):
                display_tree_with_file_selection(value, os.path.join(path, key), local_path)
        else:
            # It's a file
            file_path = os.path.join(local_path, path, key)
            if st.button(f"📄 {key}", key=f"file_{file_path}"):
                st.session_state.selected_file = file_path

def show_repo_page():
    """Render repository view page with side-by-side layout"""
    if not st.session_state.repo_data:
        st.session_state.page = 'home'
        return
    
    # Create two main columns for the layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Always show FOLDER CONTENT on the left
        st.markdown("### FOLDER CONTENT")
        with st.container():
            local_path = st.session_state.repo_data["local_path"]
            if local_path:
                tree = build_tree(local_path)
                display_tree_with_file_selection(tree, "", local_path)
            else:
                st.write("No local path found.")
    
    with col2:
        # Tabs for ISSUES, CODE VIEWER, and CHAT BOT
        tab1, tab2, tab3 = st.tabs(["ISSUES", "CODE VIEWER", "CHAT BOT"])
        
        with tab1:
            st.markdown("### REPOSITORY ISSUES")
            
            # Add search container
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text_input("Search issues by title, content, or issue number", 
                                  key="issue_search",
                                  value=st.session_state.issue_search)
                with col2:
                    if st.button("Search", key="search_button"):
                        search_issues()
            
            # Display issues with improved styling
            if st.session_state.filtered_issues:
                for issue in st.session_state.filtered_issues:
                    # Create a container for each issue
                    with st.container():
                        # Display issue number in red and title
                        issue_text = f"#{issue['number']} - {issue['title']}"
                        
                        # Use a regular button with custom styling for issues
                        if st.button(issue_text, key=f"issue_{issue['number']}"):
                            select_issue(issue)
                        
                        # Display issue metadata (date, state, comments)
                        st.write(f"Created: {issue.get('created_at', 'N/A')} | State: {issue.get('state', 'N/A')} | Comments: {issue.get('comments', 0)}")
                        st.markdown("---")
                
                # Display selected issue details
                if st.session_state.selected_issue:
                    st.markdown("### ISSUE DETAILS")
                    issue_num = st.session_state.selected_issue['number']
                    issue_title = st.session_state.selected_issue['title']
                    
                    # Display issue number and title
                    st.markdown(f"## #{issue_num} - {issue_title}")
                    
                    # Issue metadata
                    st.markdown(f"**Created by:** {st.session_state.selected_issue.get('user', {}).get('login', 'Unknown')}")
                    st.markdown(f"**Status:** {st.session_state.selected_issue.get('state', 'Unknown')}")
                    st.markdown(f"**Created at:** {st.session_state.selected_issue.get('created_at', 'Unknown')}")
                    st.markdown(f"**Updated at:** {st.session_state.selected_issue.get('updated_at', 'Unknown')}")
                    
                    # Issue body
                    st.markdown("### Description")
                    st.markdown(st.session_state.selected_issue.get('body', 'No description provided.'))
            else:
                if st.session_state.issue_search:
                    st.info(f"No issues found matching '{st.session_state.issue_search}'. Clear search to view all issues.")
                else:
                    st.info("No issues found for this repository.")
        
        with tab2:
            st.markdown("### CODE VIEWER")
            if st.session_state.selected_file:
                # Show the selected file name
                file_name = os.path.basename(st.session_state.selected_file)
                st.markdown(f"**Viewing: {file_name}**")
                
                # Display file content
                file_content = read_file_contents(st.session_state.selected_file)
                
                # Determine language for syntax highlighting
                extension = os.path.splitext(file_name)[1].lower()
                language = 'python' if extension == '.py' else 'javascript' if extension in ['.js', '.json'] else 'text'
                
                st.code(file_content, language=language)
            else:
                st.info("Select a file from the folder tree to view its contents.")
        
        with tab3:
            st.markdown("### CHAT BOT")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_messages:
                    align = "right" if msg['is_user'] else "left"
                    bg_color = "#32475b" if msg['is_user'] else "#2c2c2c"
                    st.markdown(
                        f"<div style='text-align: {align};'>"
                        f"<div style='display: inline-block; background-color: {bg_color}; padding: 8px 12px; border-radius: 15px; margin: 5px 0;'>"
                        f"<small>{msg['timestamp']}</small><br>"
                        f"{msg['text']}</div></div>",
                        unsafe_allow_html=True
                    )
            
            # Chat input area at the bottom
            user_input = st.text_input("input", key="chat_input", label_visibility="collapsed")
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("→", key="send_chat"):
                    if user_input.strip():
                        add_chat_message(user_input)
                        
                        # Here you can integrate with your backend for LLM responses
                        try:
                            # Example of calling your backend for LLM response
                            # Change the endpoint as needed
                            llm_response = requests.post(
                                "http://127.0.0.1:3000/chat/", 
                                json={
                                    "message": user_input,
                                    "repo_path": st.session_state.repo_data["local_path"],
                                    "selected_file": st.session_state.selected_file,
                                    "selected_issue": st.session_state.selected_issue
                                }
                            )
                            
                            if llm_response.status_code == 200:
                                bot_message = llm_response.json().get("response", "No response from server")
                            else:
                                bot_message = "Error communicating with the server."
                        except Exception as e:
                            bot_message = f"Server error: {str(e)}"
                            
                        add_chat_message(bot_message, is_user=False)
                        
                        # Clear input
                        st.session_state.chat_input = ""
                        st.experimental_rerun()

# Main app logic
def main():
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'repo_view':
        show_repo_page()

if __name__ == "__main__":
    main()
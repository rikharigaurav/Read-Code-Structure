import streamlit as st
import requests
import os
import git
from github import Github
import json
from datetime import datetime

# Configure Streamlit page with dark theme
st.set_page_config(page_title="GitHub Code Parser", layout="wide")

# Apply custom CSS for dark theme with improved padding and scrolling
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
    /* Improved container styling with better padding */
    .folder-tree, .code-viewer, .chat-bot, .issue-container {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 15px;
    }
    /* Custom scrollable containers */
    .scrollable-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 4px;
    }
    /* Improved file item styling */
    .file-item {
        padding: 5px;
        border-radius: 4px;
        margin-bottom: 2px;
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
        padding: 15px;
        margin-bottom: 10px;
    }
    .issue-number {
        color: #990000;
        font-weight: bold;
    }
    .search-container {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .issue-btn {
        text-align: left;
        background-color: #1e1e1e;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 12px;
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
    /* Chat message styling with improved padding */
    .chat-message {
        padding: 12px;
        margin: 8px 0;
        border-radius: 15px;
    }
    .user-message {
        background-color: #32475b;
        text-align: right;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #2c2c2c;
        text-align: left;
        margin-right: 20%;
    }
    /* Custom scrollbar for webkit browsers */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    /* Folder tree styles */
    .tree-item {
        padding: 5px;
        border-radius: 4px;
        margin: 2px 0;
    }
    .tree-folder {
        cursor: pointer;
        font-weight: bold;
    }
    .tree-file {
        cursor: pointer;
    }
    .tree-folder:hover, .tree-file:hover {
        background-color: #333333;
    }
    .tree-content {
        margin-left: 15px;
        border-left: 1px solid #444;
        padding-left: 10px;
    }
    .col1, .col2, .col3 {
        padding: 10px;
        border-radius: 4px;
        background-color: #1e1e1e;
    }

    .folder-container {
        border: 1px solid #444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        background-color: #1e1e1e;
    }

    .issue-list-container {
        border: 1px solid #444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        background-color: #1e1e1e;
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
if 'folder_state' not in st.session_state:
    st.session_state.folder_state = {}  # To track expanded/collapsed folder states

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

def add_chat_message(text, is_user=True, result_obj=None):
    """
    Add a message to the chat history
    
    Parameters:
    - text (str): The message text
    - is_user (bool): Whether the message is from the user or bot
    - result_obj (FinalResult, optional): Structured result object for bot responses
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if is_user:
        # For user messages, just add the text
        st.session_state.chat_messages.append({
            'text': text,
            'is_user': True,
            'timestamp': timestamp
        })
    else:
        if result_obj:
            # For bot messages with structured data, store each field separately
            st.session_state.chat_messages.append({
                'is_user': False,
                'timestamp': timestamp,
                'knowledge': result_obj.knowledge,
                'insights': result_obj.insights,
                'code': result_obj.code,
                'is_structured': True
            })
        else:
            # For regular bot messages without structured data
            st.session_state.chat_messages.append({
                'text': text,
                'is_user': False,
                'timestamp': timestamp,
                'is_structured': False
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

def toggle_folder(folder_path):
    """Toggle folder expanded/collapsed state"""
    if folder_path in st.session_state.folder_state:
        st.session_state.folder_state[folder_path] = not st.session_state.folder_state[folder_path]
    else:
        st.session_state.folder_state[folder_path] = True

def show_home_page():
    """Render home page"""
    st.title("GITHUB CODE PARSER")
    
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("PASTE YOUR GITHUB REPO LINK", key="repo_url", 
                     placeholder="e.g., https://github.com/username/repo")
        st.button("SUBMIT", on_click=handle_submit)

def display_tree_recursive(tree, path="", local_path="", indent_level=0):
    """
    Display folder tree with collapsible folders and clickable files
    """
    # Sort items to display folders first, then files alphabetically
    sorted_items = sorted(tree.items(), key=lambda x: (x[1] is None, x[0]))
    
    for key, value in sorted_items:
        full_path = os.path.join(path, key)
        file_path = os.path.join(local_path, full_path)
        
        if value is not None:  # It's a folder
            folder_id = f"folder_{full_path.replace(os.sep, '_').replace(' ', '_')}"
            is_expanded = st.session_state.folder_state.get(full_path, False)
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown(f"{'📂' if is_expanded else '📁'}")
            with col2:
                if st.button(f"{key}", key=folder_id, use_container_width=True):
                    toggle_folder(full_path)
            
            if is_expanded:
                # Create indented container for nested content
                with st.container():
                    st.markdown('<div class="tree-content">', unsafe_allow_html=True)
                    display_tree_recursive(value, full_path, local_path, indent_level + 1)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:  # It's a file
            file_id = f"file_{file_path.replace(os.sep, '_').replace(' ', '_')}"
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("📄")
            with col2:
                if st.button(f"{key}", key=file_id, use_container_width=True):
                    st.session_state.selected_file = file_path

def show_repo_page():
    """Render repository view page with side-by-side layout and scrollable containers with fixed heights"""
    if not st.session_state.repo_data:
        st.session_state.page = 'home'
        return
    
    # Custom CSS for fixed-height scrollable containers
    st.markdown("""
    <style>
        .fixed-height-container {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 10px;
            background-color: #0e1117;
            margin-bottom: 15px;
        }
        
        .folder-container {
            height: 700px;
            overflow-y: auto;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 10px;
            background-color: #0e1117;
        }
        
        .issue-card {
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #444;
            border-radius: 5px;
            background-color: #1e1e1e;
        }
        
        .chat-message {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        
        .user-message {
            background-color: #2e4057;
            text-align: right;
            margin-left: 20%;
        }
        
        .bot-message {
            background-color: #1e1e1e;
            text-align: left;
            margin-right: 20%;
        }
        
        .search-container {
            padding: 10px;
            margin-bottom: 15px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create two main columns for the layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### FOLDER CONTENT")
        # Create container with proper styling
        with st.container():
            st.markdown('<div class="folder-container">', unsafe_allow_html=True)
            local_path = st.session_state.repo_data["local_path"]
            if local_path:
                tree = build_tree(local_path)
                display_tree_recursive(tree, "", local_path)
            else:
                st.write("No local path found.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Tabs for ISSUES, CODE VIEWER, and CHAT BOT
        tab1, tab2, tab3 = st.tabs(["ISSUES", "CODE VIEWER", "CHAT BOT"])
        
        with tab1:
            st.markdown("### REPOSITORY ISSUES")
            
            # Add search container with improved padding
            with st.container():
                st.markdown('<div class="search-container">', unsafe_allow_html=True)
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text_input("Search issues by title, content, or issue number", 
                                    key="issue_search",
                                    value=st.session_state.issue_search)
                with col2:
                    if st.button("Search", key="search_button"):
                        search_issues()
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Use two separate columns for issues list and selected issue details
            issue_col, detail_col = st.columns([1, 1])
            
            with issue_col:
                st.markdown("### ISSUES LIST")
                # Create container with proper styling
                with st.container():
                    st.markdown('<div class="issue-list-container">', unsafe_allow_html=True)
                    
                    if st.session_state.filtered_issues:
                        for issue in st.session_state.filtered_issues:
                            # Create a container for each issue with better padding
                            st.markdown('<div class="issue-card">', unsafe_allow_html=True)
                            
                            # Display issue number in red and title
                            issue_text = f"#{issue['number']} - {issue['title']}"
                            
                            # Use a regular button with custom styling for issues
                            if st.button(issue_text, key=f"issue_{issue['number']}", use_container_width=True):
                                select_issue(issue)
                            
                            # Display issue metadata (date, state, comments)
                            st.write(f"Created: {issue.get('created_at', 'N/A')} | State: {issue.get('state', 'N/A')} | Comments: {issue.get('comments', 0)}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No issues found.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with detail_col:
                # Create fixed-height scrollable container for issue details
                st.markdown('<div class="fixed-height-container">', unsafe_allow_html=True)
                
                if st.session_state.selected_issue:
                    st.markdown("### ISSUE DETAILS")
                    issue_num = st.session_state.selected_issue['number']
                    issue_title = st.session_state.selected_issue['title']
                    issue_description = st.session_state.selected_issue.get('body', 'No description provided.')
                    
                    # Display issue number and title
                    st.markdown(f"## #{issue_num} - {issue_title}")
                    
                    # Issue metadata with improved spacing
                    st.markdown(f"**Created by:** {st.session_state.selected_issue.get('user', {}).get('login', 'Unknown')}")
                    st.markdown(f"**Status:** {st.session_state.selected_issue.get('state', 'Unknown')}")
                    st.markdown(f"**Created at:** {st.session_state.selected_issue.get('created_at', 'Unknown')}")
                    st.markdown(f"**Updated at:** {st.session_state.selected_issue.get('updated_at', 'Unknown')}")
                    st.markdown("### Description")
                    st.markdown(issue_description)
                    
                    if st.button("Ask Chatbot About This Issue"):
                        issue_prompt = f"{issue_title}: {issue_description}"
                        add_chat_message(issue_prompt)
                        try:
                            llm_response = requests.post(
                                "http://127.0.0.1:3000/chat/", 
                                json={
                                    "query": issue_prompt
                                }
                            )
                            
                            if llm_response.status_code == 200:
                                response_data = llm_response.json()
                                
                                # Check if response has structured format
                                if isinstance(response_data.get("response"), dict) and all(key in response_data["response"] for key in ["knowledge", "insights", "code"]):
                                    # Create FinalResult object
                                    from pydantic import BaseModel, Field
                                    
                                    class FinalResult(BaseModel):
                                        knowledge: str = Field(description="the knowledge for the given query")
                                        insights: str = Field(description="the insights for the given query")
                                        code: str = Field(description="the code for the given query")
                                    
                                    result_obj = FinalResult(
                                        knowledge=response_data["response"]["knowledge"],
                                        insights=response_data["response"]["insights"],
                                        code=response_data["response"]["code"]
                                    )
                                    
                                    # Add structured message
                                    add_chat_message("", is_user=False, result_obj=result_obj)
                                else:
                                    # Handle regular text response
                                    bot_message = response_data.get("response", "No response from server")
                                    add_chat_message(bot_message, is_user=False)
                            else:
                                bot_message = f"Error communicating with the server. Status code: {llm_response.status_code}"
                                add_chat_message(bot_message, is_user=False)
                        except Exception as e:
                            bot_message = f"Server error: {str(e)}"
                            add_chat_message(bot_message, is_user=False)
                        
                        # Switch to chat tab after sending message
                        st.session_state.active_tab = 2
                        st.experimental_rerun()
                else:
                    st.info("Select an issue to view details.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### CODE VIEWER")
            # Create container with proper styling
            with st.container():
                st.markdown('<div class="code-container">', unsafe_allow_html=True)
                
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
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### CHAT BOT")
            
            # Create fixed-height scrollable container for chat history
            chat_container = st.container()
            with chat_container:
                st.markdown('<div class="fixed-height-container" id="chat-history">', unsafe_allow_html=True)
                
                for msg in st.session_state.chat_messages:
                    if msg['is_user']:
                        # User message (right-aligned) with improved styling
                        st.markdown(
                            f"<div class='chat-message user-message'>"
                            f"<small>{msg['timestamp']}</small><br>"
                            f"{msg['text']}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        # Bot message (left-aligned) with improved styling
                        if msg.get('is_structured', False):
                            # Display structured response in separate boxes
                            st.markdown(
                                f"<div class='chat-message bot-message'>"
                                f"<small>{msg['timestamp']}</small>",
                                unsafe_allow_html=True
                            )
                            
                            # Knowledge box
                            with st.expander("Knowledge", expanded=True):
                                st.markdown(
                                    f"<div style='background-color: #1e3a5f; padding: 12px; border-radius: 8px;'>{msg['knowledge']}</div>",
                                    unsafe_allow_html=True
                                )
                            
                            # Insights box
                            with st.expander("Insights", expanded=True):
                                st.markdown(
                                    f"<div style='background-color: #3a5f1e; padding: 12px; border-radius: 8px;'>{msg['insights']}</div>",
                                    unsafe_allow_html=True
                                )
                            
                            # Code box
                            with st.expander("Code", expanded=True):
                                st.code(msg['code'], language="python")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            # Regular bot message
                            st.markdown(
                                f"<div class='chat-message bot-message'>"
                                f"<small>{msg['timestamp']}</small><br>"
                                f"{msg['text']}</div>",
                                unsafe_allow_html=True
                            )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Auto scroll to bottom of chat using JavaScript
                st.markdown("""
                <script>
                    function scrollToBottom() {
                        const chatHistory = document.getElementById('chat-history');
                        if (chatHistory) {
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    }
                    // Call immediately and after a slight delay to ensure content is loaded
                    scrollToBottom();
                    setTimeout(scrollToBottom, 100);
                </script>
                """, unsafe_allow_html=True)
            
            # Chat input area at the bottom with improved styling
            st.markdown('<div style="background-color: #1e1e1e; padding: 15px; border-radius: 4px;">', unsafe_allow_html=True)
            user_input = st.text_input("input", key="chat_input", label_visibility="collapsed", placeholder="Type your message here...")
            col1, col2 = st.columns([6, 1])
            with col2:
                # Send button
                if st.button("→", key="send_chat"):
                    if user_input.strip():
                        add_chat_message(user_input)
                        try:
                            # Call your backend for LLM response
                            llm_response = requests.post(
                                "http://127.0.0.1:3000/chat/", 
                                json={
                                    "query": user_input
                                }
                            )
                            
                            if llm_response.status_code == 200:
                                response_data = llm_response.json()
                                
                                # Check if response has structured format
                                if isinstance(response_data.get("response"), dict) and all(key in response_data["response"] for key in ["knowledge", "insights", "code"]):
                                    # Create FinalResult object
                                    from pydantic import BaseModel, Field
                                    
                                    class FinalResult(BaseModel):
                                        knowledge: str = Field(description="the knowledge for the given query")
                                        insights: str = Field(description="the insights for the given query")
                                        code: str = Field(description="the code for the given query")
                                    
                                    result_obj = FinalResult(
                                        knowledge=response_data["response"]["knowledge"],
                                        insights=response_data["response"]["insights"],
                                        code=response_data["response"]["code"]
                                    )
                                    
                                    # Add structured message
                                    add_chat_message("", is_user=False, result_obj=result_obj)
                                else:
                                    # Handle regular text response
                                    bot_message = response_data.get("response", "No response from server")
                                    add_chat_message(bot_message, is_user=False)
                            else:
                                bot_message = f"Error communicating with the server. Status code: {llm_response.status_code}"
                                add_chat_message(bot_message, is_user=False)
                        except Exception as e:
                            bot_message = f"Server error: {str(e)}"
                            add_chat_message(bot_message, is_user=False)
                        
                        # Clear input
                        st.session_state.chat_input = ""
                        st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Main app logic
def main():
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'repo_view':
        show_repo_page()

if __name__ == "__main__":
    main()
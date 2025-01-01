import os
from pathlib import Path
from git import Repo

def get_repo_name(repo_url: str) -> str:
    return repo_url.rstrip('/').split('/')[-1]

async def clone_repository(repo_url: str, parent_directory: str):
    try:
        repo_name = get_repo_name(repo_url)
        repo_path = os.path.join(parent_directory, repo_name)
        full_path = os.path.abspath(repo_path)
        
        print(f"Repository cloned to {full_path}")
        Path(full_path).mkdir(parents=True, exist_ok=True)
        Repo.clone_from(repo_url, full_path)
        return full_path
    
    except Exception as error:
        print(f"Failed to clone repository: {error}")

# Usage (in a real async function context)
# await clone_repository('https://github.com/user/repo.git', '/path/to/parent/dir')

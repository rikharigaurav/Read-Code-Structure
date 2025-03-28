from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.cloneRepo import clone_repository
from pathlib import Path
from utils.L_graph import getFilesContext
from utils.query import process_query
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Ensure WebSocket headers are allowed
)


class DirectoryItem(BaseModel):
    name: str
    type: str
    path: str

class data(BaseModel):
    repo_url: str    
    # docsUrl: str | None = None
class query(BaseModel):
    query: str

@app.post('/chat/')
async def chat(Body: query):
    query = Body.query
    # print(query)
    response = process_query(query)
    print(response)
    return {
        'status_code': 200,
        'response': response
    }

@app.post("/github/")
async def repo(Body: data):
    repo_url = Body.repo_url
    # docsUrl = Body.docsUrl
    print(repo_url)

    if(repo_url):
        # print("Current repo url",repoUrl),
        # Clone this github repo to system
        # fullPath = await clone_repository(repo_url, "./")
        fullPath = 'Read-Code-Structure.git'
        # print("Full system path for repo", fullPath)
        # Create a pinecone index
        repoName = None
        if(fullPath):
            repoName = Path(fullPath).name
            print(repoName)
            # await getFilesContext(fullPath, repoName)
        return {
            'status_code': 200,
            'localPath': repoName,
        }

        # Get context for all the files and code bases using Langchain and Hugginface
    else:
        return {"repoUrl": "No repo url provided"}

@app.get("/directory")
async def get_directory(path: str = None):
    try:
        # Validate and sanitize the path
        print(path)
        if not path:
            path = os.getcwd()
            
        # Security check to prevent directory traversal
        if not os.path.abspath(path).startswith(os.getcwd()):
            raise HTTPException(status_code=400, detail="Invalid path")
            
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail="Path not found")
            
        if not path_obj.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        items = []
        for item in path_obj.iterdir():
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.absolute())
            })
            
        return items
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/file")
async def read_file(path: str):
    try:
        # print(path)
        with open(path, 'r') as file:
            return file.read()  
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
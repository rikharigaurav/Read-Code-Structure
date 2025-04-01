from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.cloneRepo import clone_repository
from pathlib import Path
from typing import Dict, Any
from utils.query import create_issue_solver
from fastapi.middleware.cors import CORSMiddleware

import os
import json

app = FastAPI(title="Issue Solver API")
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
class QueryRequest(BaseModel):
    query: str

# Define the response model
class QueryResponse(BaseModel):
    status_code: int
    response: Dict[str, Any]

# Initialize the issue_solver at startup
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index = os.environ.get("PINECONE_INDEX")

# Create the issue solver instance
issue_solver = create_issue_solver(
    neo4j_uri=neo4j_uri,
    neo4j_username=neo4j_username,
    neo4j_password=neo4j_password,
    pinecone_api_key=pinecone_api_key,
    pinecone_index=pinecone_index
)

@app.post('/chat/', response_model=QueryResponse)
async def chat(body: QueryRequest):
    try:
        # Extract query from request body
        query = body.query
        
        # Call the issue solver with the query
        response = issue_solver(query)
        
        # Log the response if needed
        print(json.dumps(response.dict(), indent=2))
        
        # Return successful response
        return {
            'status_code': 200,
            'response': response
        }
    except Exception as e:
        # Log the error
        print(f"Error processing query: {str(e)}")
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
    
@app.post("/github/")
async def repo(Body: data):
    repo_url = Body.repo_url
    # docsUrl = Body.docsUrl
    print(repo_url)

    if(repo_url):
        # print("Current repo url",repoUrl),
        # Clone this github repo to system
        fullPath = await clone_repository(repo_url, "./")
        # fullPath = 'Read-Code-Structure.git'
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
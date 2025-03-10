from fastapi import FastAPI
from pydantic import BaseModel
from utils.cloneRepo import clone_repository
from pathlib import Path
from utils.L_graph import getFilesContext
from utils.query import process_query

app = FastAPI()


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
        fullPath = 'rich.git'
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

    
from fastapi import FastAPI
from pydantic import BaseModel
from utils.cloneRepo import clone_repository
from pathlib import Path
from utils.L_graph import getFilesContext
# from utils.llmCalls import llm_check

app = FastAPI()
class data(BaseModel):
    repoUrl: str    
    docsUrl: str | None = None

@app.post("/github/")
async def repo(Body: data):
    repoUrl = Body.repoUrl
    docsUrl = Body.docsUrl

    if(repoUrl):
        # print("Current repo url",repoUrl),
        # Clone this github repo to system
        # fullPath = await clone_repository(repoUrl, "./")
        fullPath = 'Read-Code-Structure'
        # print("Full system path for repo", fullPath)
        # Create a pinecone index
        if(fullPath):
            repoName = Path(fullPath).name
            print(repoName)
            # await getFilesContext(fullPath, repoName)
            # await setFileParser('./test.py', "py")
            await getFilesContext(fullPath, repoName)

        # Get context for all the files and code bases using Langchain and Hugginface

    else:
        return {"repoUrl": "No repo url provided"}
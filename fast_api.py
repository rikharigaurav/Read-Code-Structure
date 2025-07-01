from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.cloneRepo import clone_repository
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional, List
from utils.query import create_issue_solver, SolutionOutput
from fastapi.middleware.cors import CORSMiddleware
from utils.L_graph import getFilesContext
from utils.generate_readme import summarizer, FolderSummarizer


import os
import json

class SolutionOutput(BaseModel):
    summary: str
    procedural_knowledge: Optional[list[str]]
    code_solution: Optional[str]
    visualization_query: Optional[str] = None

class DirectoryItem(BaseModel):
    name: str
    type: str
    path: str

class SummaryData(BaseModel):
    localRepoPath: str


class QueryData(BaseModel):
    repo_url: str
    # docsUrl: str | None = None

class QueryRequest(BaseModel):
    query: str

class ResponseWrapper(BaseModel):
    status_code: int
    response: SolutionOutput 

app = FastAPI(title="Issue Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index = os.environ.get("PINECONE_INDEX")


issue_solver = create_issue_solver(
    neo4j_uri=neo4j_uri,
    neo4j_username=neo4j_username,
    neo4j_password=neo4j_password
    # pinecone_api_key=pinecone_api_key,
    # pinecone_index=pinecone_index
)

@app.post('/chat/', response_model=ResponseWrapper)
async def chat(body: QueryRequest):
    try:
        query = body.query
        response = issue_solver(query)
        print("Query received:_________________________", response)
        # print("Response from issue solver:")
        # print(response)
        print(json.dumps(response, indent=2))
        
        return ResponseWrapper(
            status_code=200,
            response=SolutionOutput(**response) 
        )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@app.post("/summarize-folder/")
async def summarize_folder(body: SummaryData) -> Dict[str, Any]:
    """
    Summarize a folder and return the results.
    
    Returns:
        Dictionary containing the summary results
    """

    folder_path = body.localRepoPath
    print(f"the body for summary is {body}")
    print(f"local file path {folder_path}")  # Default folder path
    
    # Validate folder path
    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Folder path '{folder_path}' does not exist"
        )
    
    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Path '{folder_path}' is not a directory"
        )
    
    try:
        # Traverse and summarize the folder
        result = await summarizer.traverse_and_summarize(folder_path)
        
        # Debug: Print the result structure and types
        print(f"DEBUG: result type: {type(result)}")
        print(f"DEBUG: result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict) and 'summary' in result:
            summary_data = result['summary']
            print(f"DEBUG: summary_data type: {type(summary_data)}")
            print(f"DEBUG: summary_data keys: {summary_data.keys() if isinstance(summary_data, dict) else 'Not a dict'}")
            
            # Safely extract data with defaults
            if isinstance(summary_data, dict):
                root_summary = summary_data.get('summary', 'No summary available')
                file_count = summary_data.get('file_count', 0)
                subfolder_count = summary_data.get('subfolder_count', 0)
                folder_name = summary_data.get('name', 'Unknown')
            else:
                # If summary_data is not a dict, treat it as the summary string
                root_summary = str(summary_data)
                file_count = 0
                subfolder_count = 0
                folder_name = 'Unknown'
        else:
            # Fallback if result structure is unexpected
            root_summary = str(result)
            file_count = 0
            subfolder_count = 0
            folder_name = 'Unknown'
        
        # Prepare response data
        response_data = {
            "folder_path": folder_path,
            "root_summary": root_summary,
            "file_count": file_count,
            "subfolder_count": subfolder_count,
            "folder_name": folder_name,
            "status": "success"
        }
        
        return response_data
        
    except Exception as e:
        # More detailed error information
        import traceback
        error_details = {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"ERROR DETAILS: {error_details}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing folder: {str(e)}"
        )
    
@app.post("/github/")
async def repo(Body: QueryData):
    repo_url = Body.repo_url
    # docsUrl = Body.docsUrl
    print(repo_url)

    if(repo_url):
        fullPath = await clone_repository(repo_url, "./")
        print("Full system path for repo", fullPath)
        # Create a pinecone index
        repoName = None
        if(fullPath):
            repoName = Path(fullPath).name
            print(repoName)
            await getFilesContext(fullPath, repoName)
        
        # Add 10 second sleep before returning
        # await asyncio.sleep(10)
        
        return {
            'status_code': 200,
            'localPath': repoName,
        }

    else:
        # Add 10 second sleep before returning error response too
        await asyncio.sleep(10)
        return {"repoUrl": "No repo url provided"}
    
@app.get("/directory")
async def get_directory(path: str = None):
    try:
          
        print(path)
        if not path:
            path = os.getcwd()
            

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
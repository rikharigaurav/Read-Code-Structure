import os
from utils.llmCalls import get_file_type
from utils.pending_rela import pending_rels
from pathlib import Path
from utils.pinecone_db import pinecone
from utils.neodb import app

async def getFilesContext(startPath: str, reponame: str):
    pending_rels.clear_relationships()
    pinecone.delete_index()
    app.remove_all()
    print(f"The repo name is {reponame}")
    ignored_files = []  
    stack = [(startPath, False)]  

    repo_id = f"FOLDER: {startPath}"
    app.create_folder_node(repo_id, reponame, startPath)
    print(stack)
    while stack:
        currentDir, processed = stack.pop()
        print(currentDir, processed)
        parent_node_id = f"FOLDER: {currentDir}"

        if processed:
            # For future creating context for folders too
            incommmingNodes = app.get_incoming_nodes(parent_node_id)
            print(f"Finished processing directory: {currentDir}")
            print(f"All the incomming nodes are listed here: {incommmingNodes}")

            print(f'parent folder = {os.path.dirname(currentDir)} \n')
            for node in incommmingNodes:
                print(f'''
                      node = {node} \n
                      ''')
                if os.path.isdir(fullPath) :
                    children_node_id = f"FOLDER: {fullPath}"
                    Type = 'FOLDER'
                else:
                    file_extension = Path(fullPath).suffix.lstrip('.')
                    children_node_id = f"FILE: {fullPath} EXT: {file_extension}"
                    Type = 'FILE'

            nameSpace = os.path.dirname(currentDir)
            print(f"Context created for {currentDir} with namespace: {nameSpace}")
            
            continue

        print(os.path.isdir(currentDir))
        if os.path.isdir(currentDir):
            print(f"Starting to open {currentDir}")
            stack.append((currentDir, True))
            
            try:  
                filesAndDirs = os.listdir(currentDir)
                print(filesAndDirs)
                fullPathList: list[str] = []
                for fileOrDir in reversed(filesAndDirs):
                    fullPath = os.path.join(currentDir, fileOrDir)
                    
                    if os.path.isdir(fullPath) and not fileOrDir.startswith("."):
                        children_node_id = f"FOLDER: {fullPath}"
                        print(children_node_id)
                        stack.append((fullPath, False))
                        app.create_folder_node(children_node_id, os.path.basename(fullPath), fullPath)
                        print("node created")
                        app.create_relation(children_node_id, parent_node_id, "BELONGS_TO")
                        print("relation created")

                        print(f"created node {children_node_id}")
                    else:
                        fullPathList.append(fullPath)

                print(fullPathList)
                await get_file_type(fullPathList, parent_node_id, reponame)

            except Exception as e:
                print(f"Error accessing directory {currentDir}: {e}")
                ignored_files.append(currentDir)
                print(ignored_files)

    pending_rels.create_all_relationships()
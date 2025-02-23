import os
from utils.llmCalls import get_file_type
from utils.pending_rela import pending_rels
from pathlib import Path

from utils.neodb import app

async def getFilesContext(startPath: str, reponame: str):
    pending_rels.clear_relationships()
    app.remove_all()
    print(f"The repo name is {reponame}")
    ignored_files = []  # List to hold files that caused errors
    stack = [(startPath, False)]  # Initialize stack with start path, False indicates it’s not fully processed

    repo_id = f"FOLDER: {startPath}"

    app.create_folder_node(repo_id, reponame, startPath)
    # create rootNode for the repoDir
    print(stack)
    while stack:
        currentDir, processed = stack.pop()
        print(currentDir, processed)
        parent_node_id = f"FOLDER: {currentDir}"

        if processed:
            # If the directory is marked as processed, we finish it here after all its children are processed
            # Collect all the context for the directed nodes and summarize for the summary
            incommmingNodes = app.get_incoming_nodes(parent_node_id)
            print(f"Finished processing directory: {currentDir}")
            print(f"All the incomming nodes are listed here: {incommmingNodes}")

            #Iterate all the incomming nodes and inserting a context summary for all the files and folder into that file and creating a context for that file 
            # TextfilePath = f"../{os.path.dirname(currentDir)}" 
            print(f'parent folder = {os.path.dirname(currentDir)} \n')
            #Create a File
            # with open(TextfilePath, "a") as file:
            for node in incommmingNodes:
                  #Extracting each nodes context and sending it to llm to summarize it.
                print(f'''
                      node = {node} \n
                      ''')
                # fullPath = os.path.join(currentDir, node)
                if os.path.isdir(fullPath) :
                    children_node_id = f"FOLDER: {fullPath}"
                    Type = 'FOLDER'
                else:
                    file_extension = Path(fullPath).suffix.lstrip('.')
                    children_node_id = f"FILE: {fullPath} EXT: {file_extension}"
                    Type = 'FILE'
                  # nodeContext =  app.get_node_context(children_node_id)
                  # file.write(f"File Path: {fullPath} | Type : {Type}  \n {nodeContext} \n\n")

            #Send this file to llm to create context
            # nameSpace = create_file_context(TextfilePath)
            nameSpace = os.path.dirname(currentDir)
            print(f"Context created for {currentDir} with namespace: {nameSpace}")
            
            #Update the Nodes Namespace Context
            # app.update_folder_context(currentDir, nameSpace)

            #Delete the file
            # os.remove(os.path.dirname(currentDir))
            continue

        print(os.path.isdir(currentDir))
        if os.path.isdir(currentDir):
            # Mark directory as processed and push it back  onto the stack
            # create a node for the directory in the graph
            print(f"Starting to open {currentDir}")
            stack.append((currentDir, True))
            
            try:  
                # List all files and directories in the current directory
                filesAndDirs = os.listdir(currentDir)
                print(filesAndDirs)
                fullPathList: list[str] = []
                # Reverse to ensure files and subdirectories are processed in correct order
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
                        # Process files immediately
                        fullPathList.append(fullPath)

                print(fullPathList)
                await get_file_type(fullPathList, parent_node_id, reponame)

            except Exception as e:
                # Log the error if there’s an issue accessing the directory and add it to ignored files
                print(f"Error accessing directory {currentDir}: {e}")
                ignored_files.append(currentDir)
                print(ignored_files)

    pending_rels.create_all_relationships()
        # else: 
            # If the current directory is not a directory, it must be a file

# app.close() 
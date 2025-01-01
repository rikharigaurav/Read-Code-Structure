import os
from utils.llmCalls import get_file_type
from utils.Fileparser import setFileParser
import asyncio
from pathlib import Path

from utils.neodb import App

importantFilesContext = {
  'README.md':"""
    The README file is your entry point into understanding the project. Here\'s how to study it:
    1. **Project Overview**: Look for a summary of what the project does, its goals, and why it exists.
    2. **Installation Instructions**: Find step-by-step instructions to set up the project on your machine.
    3. **Usage Examples**: If provided, these will show you how to use the software effectively.
    4. **Features**: Look for listed features that highlight what the project is capable of.
    5. **Contributions & Support**: Check for links or sections pointing to contributing guidelines or support options.
    6. **Next Steps**: Sometimes there are links to further documentation or tutorials to understand the project better.
    Start by scanning the table of contents, then dive deeper into each section to get a feel for the project\'s purpose, setup, and usage.
  """,

  'CONTRIBUTING.md': """
    This file outlines how you can contribute to the project. Here's how to study it:
    1. **Contribution Guidelines**: Look for rules or coding standards that contributors should follow (e.g., commit message format, coding style).
    2. **Pull Requests**: Understand how to create a pull request (PR), including any special procedures (like running tests).
    3. **Reporting Issues**: Check if there are instructions on how to report bugs or request features.
    4. **Code Reviews**: If provided, look for details on the review process (e.g., who reviews PRs, what kind of feedback to expect).
    5. **Community Interaction**: Find out how contributors should interact with the maintainers and other contributors (e.g., joining discussions).
    By thoroughly reading this file, you’ll understand the process for contributing code, documentation, or ideas to the project.
  """,

  'LICENSE': """
    The LICENSE file outlines the legal permissions and limitations associated with using the project's code. Study it by:
    1. **Usage Rights**: Determine if you're allowed to use the code for personal or commercial projects.
    2. **Modification Rights**: Check whether you’re permitted to modify and distribute the modified versions of the project.
    3. **Distribution**: Understand any restrictions on redistributing the project’s code or your modified versions.
    4. **Attribution**: Pay attention to whether you need to provide credit to the original authors.
    5. **Liability & Warranty**: Most open-source licenses disclaim liability; review this section to understand the risks.
    Understanding the license is crucial for ensuring that you're using the code in compliance with its terms.
  """,

  'CHANGELOG.md': """
    The CHANGELOG helps you track the project's development over time. Here’s how to study it:
    1. **Version History**: Look for a timeline of project versions and how the project has evolved.
    2. **Features**: Identify new features that have been added in each version.
    3. **Bug Fixes**: Check for bugs that have been fixed and if they impact any areas you're working on.
    4. **Breaking Changes**: Look for any changes that may require adjustments to your existing setup or code.
    5. **Deprecations**: Note any features that have been deprecated and what the alternatives are.
    This file will help you understand how the project is evolving and whether any changes might affect your use of the project.
""",

  'INSTALL.md': """
    This file provides instructions to get the project running on your local machine. Here's how to study it:
    1. **Prerequisites**: Look for system requirements, dependencies, or specific software versions needed.
    2. **Installation Steps**: Follow each step to set up the project environment. Pay attention to any manual setup required.
    3. **Common Issues**: Sometimes, troubleshooting tips or common installation errors are mentioned.
    4. **Environment Variables**: Check if any environment variables need to be set for the project to function.
    By carefully going through this file, you'll ensure that the project is set up properly on your machine, avoiding common pitfalls.
""",

  'CODE_OF_CONDUCT.md': """
    This file outlines the expected behavior of contributors and community members. Here’s how to study it:
    1. **Community Guidelines**: Understand the rules for respectful communication and behavior in the community.
    2. **Reporting Violations**: Check how and where to report any violations of the code of conduct.
    3. **Inclusivity**: Look for sections related to ensuring a diverse and inclusive environment.
    4. **Consequences**: See what actions are taken if someone violates the code of conduct.
    This file is essential for ensuring a healthy and welcoming community, especially for open-source projects.
""",

  'SECURITY.md': """
    This file explains how to handle security issues related to the project. Here's how to study it:
    1. **Vulnerability Reporting**: Understand how to report a security vulnerability.
    2. **Security Policies**: Review any policies around how the project handles security risks or updates.
    3. **Contact Information**: Look for email addresses or platforms for securely submitting security issues.
    4. **Disclosure Process**: Check if the project mentions how vulnerabilities are disclosed publicly after they are resolved.
    This file is critical for maintaining the project's security and should be thoroughly reviewed by anyone contributing.
""",

  'Makefile': """
    The Makefile helps you automate tasks like building, testing, and running the project. Here's how to study it:
    1. **Targets**: Look at the different tasks or "targets" (like 'build', 'test', 'run') and what commands they execute.
    2. **Dependencies**: Check what external tools or software the project depends on to run these tasks.
    3. **Common Commands**: Identify frequently used commands that contributors should know.
    4. **Customization**: Look for areas where you can modify the Makefile to fit your local development environment.
    Understanding the Makefile helps you interact with the project’s build and testing processes more efficiently.
""",

  'package.json': """
    For Node.js projects, this file contains metadata about the project. Here's how to study it:
    1. **Project Information**: Look at the \`name\`, \`version\`, and \`description\` fields to understand the project scope.
    2. **Scripts**: Check out the \`scripts\` section to find common tasks you can run (like \`npm run build\` or \`npm test\`).
    3. **Dependencies**: Look for project dependencies (in \`dependencies\` and \`devDependencies\`) to understand what tools and libraries the project uses.
    4. **Version Management**: Check if there are any version constraints on dependencies that may affect compatibility.
    By understanding this file, you get a comprehensive view of how the project is structured and what dependencies it requires.
""",

  '.gitignore': """
    The .gitignore file tells Git which files or directories to ignore in version control. Here's how to study it:
    1. **Ignored Files**: Look at the patterns to see what files (e.g., environment configs, temporary build files) should not be committed.
    2. **Project Artifacts**: Check for patterns that ignore generated files like \`node_modules/\` or \`dist/\` directories.
    3. **Local Development**: Sometimes, certain files are excluded only from local development environments, so ensure you understand why.
    4. **Custom Rules**: Check for any custom rules for excluding specific types of files that might be unique to the project.
    Understanding the \`.gitignore\` helps you avoid accidentally adding unnecessary files to the repository.
"""
}

app = App()  # Use the correct URI format
app.remove_all()

async def getFilesContext(startPath: str, reponame: str):
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
                
                # Reverse to ensure files and subdirectories are processed in correct order
                for fileOrDir in reversed(filesAndDirs):
                    fullPath = os.path.join(currentDir, fileOrDir)
                    
                    if os.path.isdir(fullPath):
                        # Push directories onto the stack to process after their children
                        children_node_id = f"FOLDER: {fullPath}"
                        print(children_node_id)
                        stack.append((fullPath, False))
                        app.create_folder_node(children_node_id, os.path.basename(fullPath), fullPath)
                        print("node created")
                        app.create_folder_relation(children_node_id, parent_node_id, "BELONGS_TO")
                        print("relation created")

                        print(f"created node {children_node_id}")
                    else:
                        # Process files immediately
                        baseName = os.path.basename(fullPath)
                        file_extension = Path(fullPath).suffix.lstrip('.')
                        children_node_id = f"FILE: {fullPath} EXT: {file_extension}"
                        print(children_node_id)
                        await get_file_type(fullPath, parent_node_id)
                        app.create_file_node(children_node_id, baseName, fullPath, file_extension)
                        app.create_folder_relation(children_node_id, parent_node_id, "BELONGS_TO")

                        print(f"created node {children_node_id}")

            except Exception as e:
                # Log the error if there’s an issue accessing the directory and add it to ignored files
                print(f"Error accessing directory {currentDir}: {e}")
                ignored_files.append(currentDir)
                print(ignored_files)

        # else: 
            # If the current directory is not a directory, it must be a file

# app.close() 
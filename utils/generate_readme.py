import os
import asyncio
import json
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from pathlib import Path
from utils.pinecone_db import pinecone
import re
from utils.neodb import app

class FolderSummarizer:
    def __init__(self):
        self.llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY_1'), model='mistral-large-latest')

    @staticmethod  # Fixed: Added staticmethod decorator
    def extract_file_path(file_string):
        # Pattern explanation:
        # [^:]+ matches one or more characters that are not colons (the file type)
        # : matches the first colon
        # ([^:]+) captures one or more characters that are not colons (the file path)
        # : matches the second colon
        pattern = r'[^:]+:([^:]+):'
        
        match = re.search(pattern, file_string)
        if match:
            file_path = match.group(1)
            # Convert forward slashes to backslashes
            return file_path.replace('/', '\\')
        return None
    
    async def traverse_and_summarize(self, root_path: str) -> Dict:
        """
        Main method to traverse folders and generate summaries
        
        Args:
            root_path: Root directory path to start traversal
            
        Returns:
            Dictionary containing the complete repository summary structure
        """
        root_path = Path(root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"Path {root_path} does not exist")
        
        namespaces = {}
        result = pinecone.get_namespace_names()
        # #print(f"DEBUG: pinecone.get_namespace_names() returned: {result}")
        # #print(f"DEBUG: type of result: {type(result)}")
        
        # Add safety check for result type
        if isinstance(result, str):
            #print(f"ERROR: Expected list/dict but got string: {result}")
            raise ValueError(f"pinecone.get_namespace_names() returned string instead of expected data structure: {result}")
        
        for namespace in result:
            #print(f"DEBUG: processing namespace: {namespace}, type: {type(namespace)}")
            
            # Fixed: Handle both string and dict formats
            if isinstance(namespace, dict) and 'name' in namespace:
                # If namespace is a dict with 'name' key
                namespace_name = namespace['name']
                file_path = self.extract_file_path(namespace_name)
                namespaces[file_path] = namespace_name
            elif isinstance(namespace, str):
                # If namespace is directly a string (which seems to be your case)
                file_path = self.extract_file_path(namespace)
                namespaces[file_path] = namespace
            else:
                print(f"WARNING: Unexpected namespace format: {namespace}, type: {type(namespace)}")
            
        #print(f"the file paths and namespaces are: {namespaces}")
        # Start depth-first traversal
        result = await self._traverse_folder_recursive(root_path, namespaces=namespaces)
        print("the result of the folder traversal is", result)
        print("\n\n")
        summary = await self.generate_repo_summary(result)
        
        print("The folder traversal and summarization is complete.")
        print(summary)
        return {
            'root_path': str(root_path),
            'summary': summary
        }
    
    async def generate_repo_summary(self, summary_data: Dict) -> Dict:
        prompt = f"""
            You are an expert software architect and technical documentation specialist. Your task is to analyze a repository structure and generate a comprehensive, professional-grade markdown document that provides deep insights into the codebase architecture, design patterns, and technical implementation.

            repository summary:
            {json.dumps(summary_data, indent=2, default=str)}
            
            Required Output Format
            Generate a complete markdown document following this exact structure:

            Overview of the Project - Provide a comprehensive introduction including the project's purpose, main functionality, architecture philosophy, and key features that define the application's core mission.
            Component Interaction Flow - Create detailed visual representations of the system architecture using markdown tables, arrows (→, ↓, ↑), flowcharts, and ASCII diagrams. Show the complete request lifecycle from user interaction through each layer (Frontend → API Layer → Service Layer → Database Layer). Use creative markdown formatting to illustrate data flow, component relationships, and system boundaries.
            Frontend Description - Analyze the user interface layer, including framework choice, component structure, state management, user experience patterns, and how the frontend communicates with backend services.
            Backend Description - Detail the server-side architecture, API design patterns, business logic implementation, data processing workflows, and integration points with external services.
            Core Components - Break down the essential modules, classes, and services that form the application's backbone, including their responsibilities, design patterns, and interdependencies.
            Technologies - Provide a comprehensive breakdown of the technology stack, including frameworks, libraries, databases, development tools, and deployment technologies with their specific roles in the application.

            Writing Quality Standards & Markdown Formatting Excellence
            Use professional technical language with precise terminology, proper heading hierarchy (H1-H6), code blocks with syntax highlighting, creative use of markdown tables and ASCII art for visual flow diagrams, bullet points and numbered lists for readability, bold/italic emphasis for key concepts, blockquotes for important architectural decisions, and emoji or symbols (→, ↓, ⚡, 🔄, 📊) to enhance visual appeal. Ensure consistent formatting throughout, include cross-references between sections, write in present tense and active voice, and provide architectural insights beyond surface-level descriptions.

            Don't add unnecessary things that are not given in context. only talk about technologies and file, folder summary that are given in the context
            Focus on providing a deep, technical understanding of the codebase architecture, design patterns, and implementation details that would be useful for developers, architects, and technical leads.
        """

        summary = self.llm.invoke([HumanMessage(content=prompt)])
        print(f"\nLLM response: {summary.content}")
        return summary.content

        

    async def _traverse_folder_recursive(self, folder_path: Path, namespaces: Dict[str, str]) -> Dict:  # Fixed: Changed type annotation
        """
        Recursively traverse folders depth-first and generate summaries
        
        Args:
            folder_path: Current folder path to process
            namespaces: Dictionary mapping file paths to namespace names
            
        Returns:
            Dictionary containing folder summary and metadata
        """
        folder_name = folder_path.name
        #print(f"Processing folder: {folder_path}")
        
        # Initialize folder data structure
        folder_data = {
            'name': folder_name,
            'path': str(folder_path),
            'files': [],
            'subfolders': {},
            'summary': '',
            'file_count': 0,
            'subfolder_count': 0
        }
        
        # Get all items in current folder
        try:
            items = list(folder_path.iterdir())
        except PermissionError:
            #print(f"Permission denied: {folder_path}")
            folder_data['summary'] = "Access denied to this folder"
            return folder_data
        
        # Separate files and folders
        files = [item for item in items if item.is_file()]
        subfolders = [item for item in items if item.is_dir() and not item.name.startswith('.')]
        
        # Process subfolders first (depth-first)
        subfolder_summaries = []
        for subfolder in subfolders:
            subfolder_summary = await self._traverse_folder_recursive(subfolder, namespaces)  # Fixed: Pass namespaces
            folder_data['subfolders'][subfolder.name] = subfolder_summary
            subfolder_summaries.append({
                'name': subfolder.name,
                'summary': subfolder_summary['summary']
            })
        
        folder_data['subfolder_count'] = len(subfolders)
        FolderID = f"FOLDER: {folder_path}"
        #print("THE FOLDER ID IS", FolderID)

        # Fixed: Added proper error handling for Neo4j query
        try:
            get_relationship = app.run_query(query=f"MATCH relationship=(childNode)-[relation]->(parentNode) RETURN childNode.id AS childNode, type(relation) as relationship, parentNode.id AS parentNode LIMIT 60;")
        except Exception as e:
            #print(f"Error querying Neo4j: {e}")
            get_relationship = []

        #print("The relationships are", get_relationship)

        # Process files in current folder
        file_contents = []
        for file_path in files:
            # Fixed: Check if file path exists in namespaces and handle errors
            file_path_str = str(file_path)
            if file_path_str in namespaces:
                namespace = namespaces[file_path_str]
                cont = "return the namespace"
                try:
                    context = pinecone.retrieve_data_from_pinecone(context=cont, score_threshold=0.1, namespace=namespace)
                    # #print(f"DEBUG: pinecone context type: {type(context)}")
                    # #print(f"DEBUG: pinecone context: {context}")
                    
                    if context and len(context) > 0:
                        # #print("the retrieved context is", context[0]['metadata'])
                        # Process each file 
                        metadata = context[0]['metadata']
                        #print(f"DEBUG: metadata type: {type(metadata)}")
                        
                        if isinstance(metadata, dict):
                            file_data = metadata.copy()
                        else:
                            #print(f"WARNING: metadata is not dict, type: {type(metadata)}, value: {metadata}")
                            file_data = {'raw_metadata': str(metadata)}
                        
                        file_data['path'] = str(file_path)

                        #print(f"Processing file: {file_path} with data: {file_data}")
                        folder_data['files'].append(str(file_path))  # Fixed: Store file path as string
                        file_contents.append(file_data)
                    else:
                        #print(f"No context found for file: {file_path}")
                        # Add basic file info even if no context
                        file_data = {
                            'name': file_path.name,
                            'path': str(file_path),
                            'type': file_path.suffix,
                            'size': file_path.stat().st_size if file_path.exists() else 0
                        }
                        folder_data['files'].append(str(file_path))
                        file_contents.append(file_data)
                except Exception as e:
                    #print(f"Error processing file {file_path}: {e}")
                    # Add basic file info on error
                    file_data = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'error': str(e)
                    }
                    folder_data['files'].append(str(file_path))
                    file_contents.append(file_data)
            else:
                #print(f"File path {file_path_str} not found in namespaces")
                # Add basic file info if not in namespaces
                file_data = {
                    'name': file_path.name,
                    'path': str(file_path),
                    'type': file_path.suffix,
                    'size': file_path.stat().st_size if file_path.exists() else 0,
                    'note': 'Not found in namespaces'
                }
                folder_data['files'].append(str(file_path))
                file_contents.append(file_data)
        
        folder_data['file_count'] = len(folder_data['files'])
        
        folder_summary = await self._generate_folder_summary(
            folder_name, file_contents, subfolder_summaries, get_relationship=get_relationship
        )
        folder_data['summary'] = folder_summary
        
        return folder_data

    async def _generate_folder_summary(self, folder_name: str, file_contents: List[Dict], subfolder_summaries: List[Dict], get_relationship) -> str:
        """
        Generate LLM-based summary for a folder
        
        Args:
            folder_name: Name of the current folder
            file_contents: List of file data dictionaries
            subfolder_summaries: List of subfolder summary dictionaries
            get_relationship: Neo4j query results
            
        Returns:
            Generated summary string
        """
        context_parts = []
        
        if subfolder_summaries:
            context_parts.append("\nSubfolder summaries:")
            for subfolder in subfolder_summaries:
                context_parts.append(f"- {subfolder['name']}: {subfolder['summary']}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are analyzing a folder within a codebase. Generate a 4-5 line concise summary that captures the folder's core purpose and architecture. Return all the main functionality from the folder and file

        FOLDER CONTEXT:
        📁 {folder_name}/ ({len(file_contents)} files, {len(subfolder_summaries)} subfolders)

        SUBFOLDERS:
        {context}

        FILE METADATA:
        {json.dumps(file_contents, indent=2, default=str)}

        RELATIONSHIP GRAPH:
        {json.dumps(get_relationship, indent=2, default=str)}

        Focus on:
        - Primary responsibility/domain of this folder
        - Critical files and their architectural roles  
        - How components interact and dependencies flow
        - Integration with subfolders (if any)

        Provide a technical summary that a developer would find immediately useful for understanding this folder's place in the overall system."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            #print(f"\nLLM response: {response}")
            # Fixed: Handle response properly - check if it's a string or has content attribute
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            #print(f"Error generating summary for {folder_name}: {str(e)}")
            return f"Folder containing {len(file_contents)} files and {len(subfolder_summaries)} subfolders"
    
    def print_summary_tree(self, summary_data: Dict, indent: int = 0) -> None:
        """
        Print the summary tree in a readable format
        
        Args:
            summary_data: The summary data structure
            indent: Current indentation level
        """
        spacing = "  " * indent
        folder_info = summary_data['summary']
        
        #print(f"{spacing}📁 {folder_info['name']}/")
        #print(f"{spacing}   Summary: {folder_info['summary']}")
        #print(f"{spacing}   Files: {folder_info['file_count']}, Subfolders: {folder_info['subfolder_count']}")
        
        # Print subfolders
        for subfolder_name, subfolder_data in folder_info['subfolders'].items():
            self.print_summary_tree({'summary': subfolder_data}, indent + 1)


# Create instance
summarizer = FolderSummarizer()
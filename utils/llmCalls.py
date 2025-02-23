import os
import json
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from utils.neodb import App
from pathlib import Path
# from utils.pinecone_db import pineconeOperation
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from typing import Union, List
from utils.pending_rela import pending_rels
from utils.tree import relative_path, read_and_parse, get_imports



app = App()
    
class FileTypeScheme(BaseModel):
    Source_Code_Files: Union[str, List[str]] = Field(
        description="Files that contain source code"
    )
    Testing_Files: Union[str, List[str]] = Field(
        description="Files that contain testing code"
    )
    Template_Files: Union[str, List[str]] = Field(
        description="Files that contain templates and markups"
    )
    Doc_Files: Union[str, List[str]] = Field(
        description="Files that contain documentation"
    )
    ignore: Union[str, List[str]] = Field(
        description="Files to ignore"
    )

file_category_handlers = {
    "Source Code Files": "process_source_code_files",  
    "Testing Files": "process_test_files",        
    "Template and Markup Files": "process_template_files",         
    "Documentation Files": "process_documentation_files", 
}

chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))

async def get_file_type(file_path_list: list[str], parentID: str, repoName: str):
    '''
        list of files will be provided and this llm call will decide to seperate the files into categories
    '''
    repoName = repoName
    print(f"file path is {file_path_list}")
    parser = JsonOutputParser(pydantic_object=FileTypeScheme)

    prompt = PromptTemplate(
        template='''
        You are tasked with categorizing a file based on its file path and inferred purpose. Follow these steps to determine the appropriate category. If the file does not belong to any predefined categories, mark it as "ignore file."

            Categories
            Source Code Files
                Files containing core logic or functionality for the project, written in known or unknown programming languages.
                Examples: Files used for frontend, backend, or scripting purposes (e.g., .js, .py, .java, etc.).

            Testing Files
                Files dedicated to writing test cases or validation logic.
                Examples: Test files for frameworks or libraries, often indicated with terms like test, spec, or feature in the filename.


            Template and Markup Files
                Files used for templates, rendering views, or designing layouts.
                Examples: Web templates, email templates, or UI-related files.

            Documentation Files
                Files providing project documentation, guides, or related information.
                Examples: Markdown or reStructuredText documentation.

            Instructions for Categorization
                Input: A query which is the file path.
                Infer the file's purpose based on its extension, or directory structure.
                Match the file to one of the categories above.
                If the file is purely a log, cache, or temporary file with no architectural or logical relevance and doesn't belong to any above category, mark it as "ignore file."
                Return the output in the specified JSON format.

            "Please return a valid JSON object with the following keys: Source_Code_Files, Testing_Files, Template_Files, Doc_Files, and ignore. Each key should map to either a string or a list of strings representing file paths. Do not include any extra text, explanations, or formatting outside of the JSON object."
            \n{format_instructions}\n{query}\n
        ''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | chat

    try:
        # Invoke the LLM and parse the result
        output = prompt_and_model.invoke({"query": file_path_list})
        result = parser.invoke(output)
        print(result)

        # Mapping from our model's field names to the category names used in the handler mapping.
        field_to_category = {
            "Source_Code_Files": "Source Code Files",
            "Testing_Files": "Testing Files",
            "Template_Files": "Template and Markup Files",
            "Doc_Files": "Documentation Files",
        }

        # Loop through each category field and call its associated handler function.
        # Debugging: Print type of result
        print("jasdfhjdh")
        print(f"Type of result: {type(result)}")

        # Check if result is a dictionary and print available keys
        if isinstance(result, dict):
            print(f"Keys in result: {list(result.keys())}")
        else:
            print(f"Attributes in result: {dir(result)}")  # If it's an object, list attributes

        # Loop through each category field and call its associated handler function.
        for field, category in field_to_category.items():
            files = result.get(field, []) if isinstance(result, dict) else getattr(result, field, [])

            if not files:
                print(f"No files found for category: {category}. Skipping processing.")
                continue

            if not isinstance(files, list):  
                files = [files]

            handler_function_name = file_category_handlers.get(category)

            # Debugging: Print handler function name
            print(f"Processing category: {category}, Handler function: {handler_function_name}")

            handler_function = globals().get(handler_function_name)

            if handler_function:
                for filePath in files:
                    try:
                        await handler_function(filePath, parentID, repoName)
                    except Exception as e:
                        print(f"Error executing {handler_function_name} for file '{filePath}': {str(e)}")
            else:
                print(f"No handler function found for category '{category}'.")


        
        # Optionally handle files that should be ignored.
        if result.ignore:
            if isinstance(result.ignore, str):
                ignore_files = [result.ignore]
            else:
                ignore_files = result.ignore
            print(f"Ignoring files: {ignore_files}")

    except ValidationError as ve:
        print(f"Validation Error: {ve}")
        result = {"file_path": file_path_list, "file_category": "unknown", "ignore": True}
    except Exception as e:
        print(f"Unexpected Error: {e}")
        result = {"file_path": file_path_list, "file_category": "error", "ignore": True}

    return 1

async def process_test_files(fullPath: str, ParentID: str, REPONAME: str):
    """
        file_id: str = Field(description="ID for the file ")
        file_path: str = Field(description="Path of the testing file")
        file_ext: str = Field(description="Extention of the file")
        file_name: str = Field(description="The name of the testing file, often includes test, spec, or feature")
        test_framework: str = Field(description="The testing framework used, such as Jest, Mocha, JUnit, or Pytest")
        test_reference_dict: dict = Field(
            description="Dictionary containing key-value pairs (str: list) of test type and reference function or class name"
        )
    """

    file_name = os.path.basename(fullPath)
    file_extension = Path(fullPath).suffix.lstrip('.')
    file_id = f"TESTINGFILE:{fullPath}:{file_extension}"
    prompt = """
            "Analyze the provided testing file and summarize its key components. Include:
            The testing framework used (e.g., Jest, Mocha, JUnit, Pytest).
            The purpose of the tests in the file (e.g., unit tests, integration tests, end-to-end tests).
            A high-level breakdown of test cases and their objectives.
            Any conditions being validated, including inputs, expected outputs, and mock data.
            Use the identified test framework to infer testing style and organize the summary accordingly."
        """
    
    # output = process_llm_calls(fullPath, prompt, 'TestingFile')
    code = None
    with open(fullPath, 'r') as f:
        code = f.read()

    test_reference_dict = get_imports(code)

    # create node for testing file and respective relations
    app.create_testing_file_node(file_id, file_name, fullPath, file_extension, 'pytest', test_reference_dict)
    app.create_relation(file_id, ParentID, "BELONGS_TO")


    for test_function, function_file_path in test_reference_dict.items():
        rel_path = relative_path.get_relative_path(function_file_path)
        if isinstance(test_function, tuple):
            if(test_function[0]):
                references_ID = f"FUNCTION:{rel_path}:{test_function[0]}"
                pending_rels.add_relationship(file_id, references_ID, 'TESTS')
        else:
            if test_function :
                references_ID = f"FUNCTION:{rel_path}:{test_function}"
                pending_rels.add_relationship(file_id, references_ID, 'TESTS')

    # namespace = pineconeOperation.load_text_to_pinecone(output['result'], REPONAME)
    # app.update_summary_context(file_id, output['prop']['summary'])
    # app.update_folder_context(file_id, namespace)

    print("node and relation has been created between test file and relative nodes")


async def process_source_code_files(file_path: str, ParentID: str, reponame: str):
    file_extension = Path(file_path).suffix.lstrip('.')
    file_name = os.path.basename(file_path)
    file_id = f"SOURCECODEFILE:{file_path}:{file_extension}"
    prompt = """
            "Analyze the provided testing file and summarize its key components. Include:
            The testing framework used (e.g., Jest, Mocha, JUnit, Pytest).
            The purpose of the tests in the file (e.g., unit tests, integration tests, end-to-end tests).
            A high-level breakdown of test cases and their objectives.
            Any conditions being validated, including inputs, expected outputs, and mock data.
            Use the identified test framework to infer testing style and organize the summary accordingly."
        """
    #connected file to its coresponding folder 
    app.create_data_file_node(file_id, file_name, file_path, file_extension)
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    # output = process_llm_calls(file_path, prompt, 'SourceCodeFile')

    res = await read_and_parse(file_path, file_id)
    if res:
        print('----------------------------> reading and parsing is done')
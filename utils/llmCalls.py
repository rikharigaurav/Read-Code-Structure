import os
import json
from pydantic import BaseModel, Field, ValidationError, model_validator
from langchain.prompts import PromptTemplate
from utils.neodb import App
from pathlib import Path
from utils.pinecone import pineconeOperation
# from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from memory import process_llm_calls
from langchain_mistralai.chat_models import ChatMistralAI

pineconeOperation = pineconeOperation()
app = App()
    
class FileTypeScheme(BaseModel):
    file_path: str = Field(description="Path of the file")
    file_category: str = Field(description="Type of the file")
    ignore: bool = Field(description="Should ignore this file or not")

    '''
        left node dictionary : {
            "Parent fileID" : {
                    "Children FileID : "",
                    "relation" : ""
                }
        }
    '''

# Dictionary mapping categories to processing functions
file_category_handlers = {
    "Source Code Files": "process_source_code_files",  
    "Testing Files": "process_test_files",        
    "Template and Markup Files": "process_template_files",          
    "Code Metadata and Graph Files": "process_metadata_files",  
    "Documentation Files": "process_documentation_files", 
}

chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
async def get_file_type(file_path: str, parentID: str, repoName: str, left_nodes : dict):
    print(f"file path is {file_path}")
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


            \n{format_instructions}\n{query}\n
        ''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | chat

    try:
        # Invoke the chain and parse the output
        output = prompt_and_model.invoke({"query": "pyread.ts"})
        result = parser.invoke(output)
        print(result.ignore)

        if result.ignore:
            return 
        
        if result.file_category in file_category_handlers:
            handler_function_name = file_category_handlers[result.file_category]
            handler_function = globals().get(handler_function_name)

            if handler_function:
                try:
                    handler_function(
                        result.file_path, 
                        parentID,         
                        repoName,
                        left_nodes
                    )
                except Exception as e:
                    print(f"Error executing {handler_function_name}: {str(e)}")
            else:
                print(f"Handler function '{handler_function_name}' not found.")
        else:
            print(f"File category '{result.file_category}' not recognized.")
            

    except ValidationError as ve:
        print(f"Validation Error: {ve}")
        result = {"file_path": file_path, "file_category": "unknown", "ignore": True}
    except Exception as e:
        print(f"Unexpected Error: {e}")
        result = {"file_path": file_path, "file_category": "error", "ignore": True}

    return result


async def get_Test_file_context(fullPath: str, ParentID: str, REPONAME: str, left_nodes : dict):
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
    file_id = f"TESTINGFILE: {fullPath} EXT: {file_extension}"


    prompt = """
            "Analyze the provided testing file and summarize its key components. Include:
            The testing framework used (e.g., Jest, Mocha, JUnit, Pytest).
            The purpose of the tests in the file (e.g., unit tests, integration tests, end-to-end tests).
            A high-level breakdown of test cases and their objectives.
            Any conditions being validated, including inputs, expected outputs, and mock data.
            Use the identified test framework to infer testing style and organize the summary accordingly."
        """
    
    output = process_llm_calls(fullPath, prompt, 'TestingFile')

    # create node for testing file and respective relations
    app.create_testing_file_node(file_id, file_name, fullPath, file_extension, output['prop']['test_framework'], output['prop']['test_reference_dict'], output['prop']['summary'])
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    test_reference_dict: dict = output['prop']['test_reference_dict']

    for test_type, references in test_reference_dict.items():
        for reference in references:
            # CHECK IF NODE EXISTS (reference node)
            reference_node = app.get_node_by_id(reference)
            if(reference_node):
                #create relation with testing file of that node
                app.create_relation(file_id, reference, test_type)
            else:
                #SAVE THAT NODE IN THE HASH MAP AND RECHECK IT AT LAST
                left_nodes['']
    namespace = pineconeOperation.load_text_to_pinecone(output['result'], REPONAME)
    app.update_summary_context(file_id, output['prop']['summary'])
    app.update_folder_context(file_id, namespace)

    print("node and relation has been created between test file and relative nodes")

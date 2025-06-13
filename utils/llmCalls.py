import os
import json
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from utils.neodb import app
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from typing import Union, List
from language.tree_python import read_and_parse_python, extract_imports
from utils.pinecone_db import pinecone
from utils.analyze_test_file import analyze_test_file
from utils.analyze_doc_file import analyze_documentation_file
from utils.analyze_template_file import analyze_markup_template

class TestType(BaseModel):
    test_framework: str = Field(
        description="Framework used by the file for test cases"
    )

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

chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY_2'))

async def get_file_type(file_path_list: list[str], parentID: str, repoName: str):
    '''
        list of files will be provided and this llm call will decide to seperate the files into categories
    '''
    repoName = repoName
    print(f"file path is {file_path_list}")
    parser = JsonOutputParser(pydantic_object=FileTypeScheme)

    prompt = PromptTemplate(
        template='''
        Categorize the provided file paths into exactly one of these categories. Analyze file extensions, directory names, and file names carefully.

        **Categories:**

        **Testing_Files:** Test cases and validation logic
        - Patterns: Contains "test", "spec", "__test__", ".test.", ".spec.", "tests/", "test/"
        - Extensions: Same as source code but in test contexts
        - Examples: tests/test_user.py, components/Header.test.jsx, spec/user_spec.rb
        - Keywords: unittest, pytest, jest, mocha, karma, cypress

        **Source_Code_Files:** Core application logic and functionality
        - Extensions: .py, .js, .ts, .jsx, .tsx, .java, .cpp, .c, .cs, .php, .rb, .go, .rs, .swift, .kt, .scala, .dart, .vue, .svelte
        - Examples: src/main.py, components/Header.jsx, utils/helper.js, models/User.java
        - Exclude: Files with "test", "spec", "__pycache__", "node_modules" in path

        **Template_Files:** UI templates and markup for rendering
        - Extensions: .html, .htm, .xml, .jinja, .jinja2, .j2, .hbs, .handlebars, .mustache, .twig, .blade.php, .erb, .ejs, .pug, .jade
        - Examples: templates/index.html, views/user.hbs, email_templates/welcome.jinja2
        - Directories: templates/, views/, layouts/

        **Doc_Files:** Documentation and guides
        - Extensions: .md, .rst, .txt, .doc, .docx, .pdf (if clearly docs)
        - Names: README, CHANGELOG, LICENSE, CONTRIBUTING, INSTALL, GUIDE, MANUAL
        - Examples: README.md, docs/api.rst, CONTRIBUTING.txt
        - Directories: docs/, documentation/

        **ignore:** Everything else
        - Config files: .json, .yml, .yaml, .toml, .ini, .cfg, .conf, .env
        - Build/Package: package.json, requirements.txt, Dockerfile, Makefile, .gitignore
        - Images/Media: .png, .jpg, .jpeg, .gif, .svg, .ico, .mp4, .mp3, .pdf (non-docs)
        - Compiled/Generated: .pyc, .class, .o, .exe, .dll, .so, .jar, .war
        - IDE/Editor: .vscode/, .idea/, .vs/, *.swp, *.swo
        - Dependencies: node_modules/, __pycache__/, .git/, venv/, env/
        - Logs/Cache: *.log, *.cache, *.tmp, .DS_Store
        - Lock files: package-lock.json, yarn.lock, Pipfile.lock

        **Rules:**
        1. One file = one category only
        2. File extension takes priority over directory
        3. Test-related keywords override source code classification
        4. When uncertain, use "ignore"
        5. Case-insensitive matching for keywords
        6. Don't mutate input file paths and return them as is

        **Examples:**
        - "src/components/Button.jsx" → Source_Code_Files
        - "tests/components/Button.test.jsx" → Testing_Files  
        - "templates/index.html" → Template_Files
        - "docs/README.md" → Doc_Files
        - "package.json" → ignore
        - "build/main.js" → ignore
        - ".env" → ignore

        Analyze these file paths: {query}

        Return valid JSON only:
        {{
            "Source_Code_Files": [],
            "Testing_Files": [],
            "Template_Files": [],
            "Doc_Files": [],
            "ignore": []
        }}

        \n{format_instructions}\n
        ''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt | chat

    try:
        # Invoke the LLM and parse the result
        output = prompt_and_model.invoke({"query": file_path_list})
        result = parser.invoke(output)
        print(result)

        field_to_category = {
            "Source_Code_Files": "Source Code Files",
            "Testing_Files": "Testing Files",
            "Template_Files": "Template and Markup Files",
            "Doc_Files": "Documentation Files",
            "ignore": "ignore"
        }
        print(f"Type of result: {type(result)}")

        if isinstance(result, dict):
            print(f"Keys in result: {list(result.keys())}")
        else:
            print(f"Attributes in result: {dir(result)}")

        for field, category in field_to_category.items():
            if(field == "ignore" and category == "ignore"):
                continue

            files = result.get(field, []) if isinstance(result, dict) else getattr(result, field, [])

            if not files:
                print(f"No files found for category: {category}. Skipping processing.")
                continue

            if not isinstance(files, list):  
                files = [files]

            handler_function_name = file_category_handlers.get(category)

            handler_function = globals().get(handler_function_name)

            if handler_function:
                for filePath in files:
                    try:
                        await handler_function(filePath, parentID, repoName)
                    except Exception as e:
                        print(f"Error executing {handler_function_name} for file '{filePath}': {str(e)}")
            else:
                print(f"No handler function found for category '{category}'.")
                
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
    code = None
    test_file_imports = None
    llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
    parser = JsonOutputParser(pydantic_object=TestType)

    with open(fullPath, 'r') as f:
        code = f.read()

    test_file_imports = extract_imports(fullPath)
    print(test_file_imports)

    prompt_template = PromptTemplate(
        template = '''
        Given the following import list from a test file, identify the testing framework used to test the functions. 
            If multiple frameworks are present, list them separated by commas; if none are detected, output "None".
            
            Import list: {test_file_imports}
            
            {format_instructions}
    ''',
        input_variables = ['test_file_imports'], 
        partial_variables = {'format_instructions': parser.get_format_instructions()},
    )

    chain = prompt_template | chat | parser
    result = chain.invoke({
        'test_file_imports': test_file_imports
    })
    print(f"The test framework used in the result is {result['test_framework']}")

    analysis_result = analyze_test_file(
        test_content=code,
        test_file_id=file_id,
        llm=llm,
        import_statements=test_file_imports,
    )

    print(f"Analysis result: {analysis_result}")
    
    all_test_cases = analysis_result["all_test_cases"]
    metadata = analysis_result["metadata"]
    
    # print(f"Test cases: {all_test_cases}")
    # print(f"Metadata: {metadata['file_purpose']}")

    app.create_testing_file_node(file_id, file_name, fullPath, file_extension, test_framework=result['test_framework'], summary=metadata['file_purpose'])
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    pinecone.load_text_to_pinecone(file_path = fullPath, file_id= file_id, metadata=metadata)
    # print("node and relation has been created between test file and relative nodes")


async def process_source_code_files(file_path: str, ParentID: str, reponame: str):
    file_extension = Path(file_path).suffix.lstrip('.')
    file_name = os.path.basename(file_path)
    file_id = f"SOURCECODEFILE:{file_path}:{file_extension}"

    app.create_data_file_node(file_id, file_name, file_path, file_extension)
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    file_metadata = await read_and_parse_python(file_path, file_id)

    #delete it 
    # await read_and_parse_python(file_path, file_id)

    app.update_node_summary(node_id = file_id, summary = file_metadata['summary'])

    pinecone.load_text_to_pinecone(file_path = file_path, file_id= file_id, metadata= file_metadata)

async def process_documentation_files(file_path: str, ParentID: str, reponame: str):
    file_extension = Path(file_path).suffix.lstrip('.')
    file_name = os.path.basename(file_path)
    file_id = f"DOCUMENTATIONFILE:{file_path}:{file_extension}"
    llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        file_content = f.read()

    metadata = analyze_documentation_file(file_content, llm=llm)
    print(f"Metadata: {metadata}")

    app.create_documentation_file_node(file_id, file_name, file_path, file_extension, summary=metadata['summary'])
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    pinecone.load_text_to_pinecone(file_path = file_path, file_id= file_id, metadata= metadata)

async def process_template_files(file_path: str, ParentID: str, reponame: str):
    file_extension = Path(file_path).suffix.lstrip('.')
    file_name = os.path.basename(file_path)
    file_id = f"TEMPLATEMARKUPFILE:{file_path}:{file_extension}"

    metadata = analyze_markup_template(file_path)

    app.create_template_markup_file_node(file_id, file_name, file_path, file_extension, summary=metadata['template_purpose'])
    app.create_relation(file_id, ParentID, "BELONGS_TO")

    pinecone.load_text_to_pinecone(file_path = file_path, file_id= file_id, metadata= metadata)
    

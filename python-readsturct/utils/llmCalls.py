import os
import json
from pydantic.v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# Hugging Face LLM initialization using HuggingFace Hub token
repo_id = "google/flan-t5-large"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
)
# prompt_template = "What is the capital of France?"

# def llm_check():
#     template = PromptTemplate(template="What is the capital of France?", input_variables=[])
#     chain = LLMChain(llm=llm, prompt=template)
#     response = chain.run()
#     print(response)


# Pydantic model for structured parsing of file analysis
class FunctionSummary(BaseModel):
    functionName: str = Field(...,description="Name of the function")
    functionSummary: str = Field(..., description="Brief summary of the function")


class FileAnalysisSchema(BaseModel):
    fileSummary: str = Field(..., description="Summary of the entire file")
    functionSummaries: list[FunctionSummary] = Field(..., description="Summaries of functions in the file")

# Read file contents, generate prompt, and analyze the file with LLM
async def llm_output_for_codebase_files(file_path: str, repo_name: str):
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()

        prompt = f"""
            You are provided with a code file located at {file_path}. Your task is to thoroughly analyze this file and provide a comprehensive understanding of its contents. This involves both summarizing the file as a whole and analyzing each function it contains to determine its specific role and purpose.

    The steps you need to follow are:

    1. **Input**: The input you are given includes the full file path ({file_path}) and the complete content of the file ({file_content}).
        - The **filePath** helps identify the location and purpose of the file within a larger codebase (if applicable).
        - The **fileContent** is the actual code or documentation that needs to be parsed, understood, and summarized.
    
    2. **Purpose**: Your task is to analyze the provided code and fulfill the following needs:
        - **Understand the overall purpose of the file**: What role does this file play in the project? Is it a configuration file, a core logic file, or a utility? Identify its primary purpose.
        - **Break down and understand each function in detail**: For each function in the file:
            - Identify the function’s name and signature.
            - Summarize what the function does in layman's terms. This summary should focus on the function's input, output, and the operations performed within.
            - If the function interacts with external libraries, APIs, or modules, briefly explain the role of those interactions.
        - **Identify key elements**: Highlight important classes, variables, constants, or any non-functional code sections (comments, type definitions, etc.) that are essential to understanding the file's behavior.

    3. **Output**: The result of your analysis should include the following:
        - **File Summary**: Provide a high-level overview of the entire file, explaining its general purpose and the context it fits into within the project.
        - **Function Summaries**: For each function in the file, include:
        - The function name.
        - A brief description of the function's purpose.
        - Details of the function's inputs (parameters) and outputs (return values).
        - Key internal logic, including any significant calculations, loops, or conditionals.
        - Any notable side effects, such as file I/O, database interactions, network requests, or state changes.
    
    Example Output:
    File Summary:

        This file handles user authentication, including validating credentials and managing sessions.
    Function Summaries:

        Function Name: validateLoginCredentials()

        Function Purpose: Validates a user's login credentials against the database.
        Inputs:
            username: string - The user's username.
            password: string - The user's password.
        Outputs: Returns a boolean indicating success or failure.
        Key Internal Logic: Checks the database for matching credentials and returns the result.
        Notable Side Effects: May log failed attempts to an external logging system.
        
        Function Name: generateAuthToken()

        Function Purpose: Creates a JWT token for authenticated users.
        Inputs:
            userData: User - An object containing user details.
        Outputs: Returns a string representing the generated token.
        Key  Internal Logic: Signs the token with a secret key and encodes user information.
        Notable Side Effects: None. 
        """

        # Create the prompt template
        # prompt = PromptTemplate.from_template(prompt_template)

        parser = PydanticOutputParser(pydantic_object=FileAnalysisSchema)
        
        prompt = PromptTemplate(
            template="\n{format_instructions}\n{prompt}\n",
            input_variables=prompt,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Run the chain
        chain = prompt | llm | parser
        # prompt_and_model = prompt | llm
        output = chain.invoke({
                "prompt": prompt  # Replace with actual prompt text
            })
        with open('log.txt', 'w') as log:
            log.write(
                f"""
                File Path: {file_path}
                    {output}
                """
            )
    
    except Exception as error:
        with open('log.txt', 'w') as log:
            log.write(
                f"""
                File Path: {file_path}
                    {error}
                """
            )
        print(f"Error in analyzing file: {error}")

# # Example for Meta Files
async def llm_output_for_meta_files(file_path: str, system_prompt: str, repo_name: str):
    try:
        # Step 1: Read the file content
        with open(file_path, 'r') as f:
            file_content = f.read()

        # Step 2: Prepare the prompt by combining the system prompt and file content
        prompt_template = f"""
        File path: {file_path}\n\n
        {system_prompt}\n\nFile Content:\n{file_content}
        """

        # Create the prompt template
        prompt = PromptTemplate.from_template(prompt_template)

        # Create and run the chain
        chain = prompt | llm 
        # chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({
            "prompt": prompt
        })

        with open('./log.txt', 'w') as log:
            log.write(
                f"""
                File Path: {file_path}
                    {result}
                """
            )
        
        print(f"File Path: {file_path}")
        print(f"LLM Output: {result}")

    except Exception as error:
        print(f"Error processing the file with LLM: {error}")
        with open('log.txt', 'w') as log:
            log.write(
                f"""
                File Path: {file_path}
                    {error}
                """
            )
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# import logging
from utils.pending_rela import pending_rels

# Configure logging
# logging.basicConfig(level=logging.INFO)
#logger = logging.get#logger(__name__)

class TestCase(BaseModel):
    node_type: str = Field(description="The tested node is function or api_endpoint")
    function_name_or_route_method: str = Field(description="Name of the function or route method being tested")
    relative_path_or_api_url: str = Field(description="Relative path to the function or API URL")

# Simplified TestFileMetadata with only 4 essential string properties
class TestFileMetadata(BaseModel):
    file_purpose: str = Field(description="Overall purpose and scope of this test file")
    test_framework_and_approach: str = Field(description="Testing framework used and overall testing methodology")
    test_coverage_summary: str = Field(description="Summary of what is being tested and coverage approach")
    key_insights: str = Field(description="Key insights, patterns, and important aspects of the test file")

class TestFileChunk(BaseModel):
    start_line: int
    end_line: int
    content: str

def analyze_test_file(
    test_content: str,
    test_file_id: str,
    max_chunk_chars: int = 4000,
    overlap_chars: int = 400,
    llm = None,
    import_statements: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze test file content to extract information about functions being tested
    and generate metadata about the test file's purpose and approach.
    
    Args:
        test_content: Content of the test file
        max_chunk_chars: Maximum characters per chunk for processing
        overlap_chars: Character overlap between chunks
        llm: Language model to use for analysis
        
    Returns:
        Dictionary containing test cases and file metadata
    """
    try:
        all_test_cases = []
        test_intuition = ""
        
        if len(test_content) < max_chunk_chars:
            # For small files, analyze in one go
            result = _analyze_test_chunk(test_content, 0, len(test_content.split('\n')), llm)
            all_test_cases.extend(result['test_cases'])
            test_intuition = result['chunk_intuition']
        else:
            chunks = _chunk_test_file(test_content, max_chunk_chars, overlap_chars)
            
            for i, chunk in enumerate(chunks):
                result = _analyze_test_chunk(
                    chunk.content, 
                    chunk.start_line, 
                    chunk.end_line,
                    llm,
                    import_statements
                )
                print(result)
                all_test_cases.extend(result['test_cases'])
                test_intuition += result['chunk_intuition'] + " "
                
            seen_functions = set()
            unique_test_cases = []
            for test_case in all_test_cases:
                if test_case['function_name_or_route_method'] not in seen_functions:
                    seen_functions.add(test_case['function_name_or_route_method'])
                    unique_test_cases.append(test_case)
                else:
                    for existing in unique_test_cases:
                        if existing['function_name_or_route_method'] == test_case['function_name_or_route_method']:
                            pass
            
            all_test_cases = unique_test_cases
                
        # Fixed the enumeration bug - enumerate returns (index, item) tuple
        for index, test_case in enumerate(all_test_cases):
            # print(f"Processing test case {index}: {test_case}")
            if test_case['node_type'] == "function":
                node = f"FUNCTION:{test_case['relative_path_or_api_url']}:{test_case['function_name_or_route_method']}"
                pending_rels.add_relationship(test_file_id, node, "TEST")

            elif test_case['node_type'] == "api_endpoint":
                node = f"APIENDPOINT:{test_case['relative_path_or_api_url']}:{test_case['function_name_or_route_method']}"
                pending_rels.add_relationship(test_file_id, node, "TEST")

        metadata = _generate_test_file_metadata(test_content, all_test_cases, test_intuition, llm)
        
        # print('metadata from the test file is', metadata)
        return {
            "all_test_cases": all_test_cases,
            "metadata": metadata
        }
    
    except Exception as e:
        #logger.error(f"Error analyzing test file: {str(e)}")
        return {
            "all_test_cases": [],
            "metadata": {
                "file_purpose": f"Error analyzing test file: {str(e)}",
                "test_framework_and_approach": "Unknown due to error",
                "test_coverage_summary": "Unknown due to error", 
                "key_insights": f"Analysis failed with error: {str(e)}"
            }
        }

def _chunk_test_file(
    test_content: str,
    max_chars: int, 
    overlap_chars: int
) -> List[TestFileChunk]:
    """
    Split test file into overlapping chunks based on character count.
    Similar to the chunking method in the original code.
    """
    lines = test_content.split('\n')
    chunks = []
    current_chunk = []
    current_chars = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_chars = len(line) + 1  
        
        if current_chars + line_chars > max_chars and current_chunk:
            chunks.append(TestFileChunk(
                start_line=start_line,
                end_line=i-1,
                content='\n'.join(current_chunk)
            ))
            
            overlap_lines = []
            overlap_chars_count = 0
            for prev_line in reversed(current_chunk):
                prev_line_chars = len(prev_line) + 1
                if overlap_chars_count + prev_line_chars <= overlap_chars:
                    overlap_lines.insert(0, prev_line)
                    overlap_chars_count += prev_line_chars
                else:
                    break
            
            current_chunk = overlap_lines
            current_chars = overlap_chars_count
            start_line = i - len(overlap_lines)
        
        current_chunk.append(line)
        current_chars += line_chars

    if current_chunk:
        chunks.append(TestFileChunk(
            start_line=start_line,
            end_line=len(lines)-1,
            content='\n'.join(current_chunk)
        ))
    
    #logger.info(f"Test file split into {len(chunks)} chunks")
    return chunks

def _analyze_test_chunk(
    chunk_content: str,
    start_line: int,
    end_line: int,
    llm,
    import_statements: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze a single chunk of test file to extract test cases and intuition.
    """
    try:
        class ChunkAnalysis(BaseModel):
            test_cases: List[TestCase] = Field(description="List of test cases found in this chunk")
            chunk_intuition: str = Field(description="Key intuition about testing approach from this chunk")
            
        parser = PydanticOutputParser(pydantic_object=ChunkAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert test code analyst with deep understanding of testing frameworks, 
            methodologies, and best practices across multiple programming languages (Python, JavaScript, TypeScript, Java, etc.). 
            You're analyzing a test file (or a chunk of a test file) to extract information about what is being tested.
            
            Your task is to identify two types of tests:
            
            1. Function tests: For each function being tested, determine:
            - The function name
            - The relative import path or package/module path (based on import/require/include statements)
            
            2. API endpoint tests: For each API endpoint being tested, determine:
            - The HTTP method (GET, POST, PUT, DELETE, etc.)
            - The API URL path (extract the full path after the base URL)
            
            CRITICAL INSTRUCTIONS FOR API ENDPOINT PATHS:
            - When you see URLs like "http://localhost:8000/api/tasks/{{task_id}}", extract "/api/tasks/<task_id>" 
            - Convert path parameters from {{param}} format to <param> format
            - Include the full path starting from the root (including /api/ prefix if present)
            - Remove the base URL (http://localhost:8000) but keep the complete path
            - Look for patterns like client.get(), client.post(), requests.get(), fetch(), etc.
            
            For each test case, clearly specify whether it's testing a function or an API endpoint by setting node_type to "function" or "api_endpoint" accordingly.
            
            Also provide a brief intuition about the overall testing approach in this chunk.
            
            Be precise and focus on the actual test code, not boilerplate or setup code.
            
            Adapt your analysis to the specific language patterns:
            - Python: Look for imports, unittest/pytest fixtures, test_ prefixed methods
            - JavaScript/TypeScript: Check for require/import statements, describe/it blocks, Jest/Mocha patterns
            - Java: Examine import statements, JUnit annotations (@Test), test method names
            - Other languages: Apply similar pattern recognition appropriate to the language"""),
            
            ("user", """
            Import statements: {import_statements}
            
            Test file chunk (Lines {start_line} to {end_line}):
            ```
            {chunk_content}
            ```
            
            Extract all test cases and provide intuition about the testing approach in this chunk.
            Pay special attention to import/require statements to determine relative paths for functions being tested.
            For API endpoints, identify the HTTP method and the COMPLETE API URL PATH.
            
            {format_instructions}
            
            Example Output for function test:
            example code for function test : 
            from app.utils.calculations import calculate_total
            def test_calculate_total():
                assert calculate_total(2, 3) == 5
            
            ```json
            {{
                "test_cases": [
                    {{
                        "node_type": "function",
                        "function_name_or_route_method": "calculate_total",
                        "relative_path_or_api_url": "app.utils.calculations"
                    }}
                ],
                "chunk_intuition": "This test file focuses on unit testing the calculation utilities."
            }}
            ```
            
            Example Output for API endpoint test:
            code for API endpoint test:
            def test_create_user(client):
                response = client.post('http://localhost:8000/api/v1/users/{{userID}}', json={{'name': 'John Doe'}})
                assert response.status_code == 201
             
            IMPORTANT: Extract the COMPLETE path after the base URL and convert {{}} to <>
            ```json
            {{
                "test_cases": [
                    {{
                        "node_type": "api_endpoint",
                        "function_name_or_route_method": "POST",
                        "relative_path_or_api_url": "/api/v1/users/<userID>"
                    }}
                ],
                "chunk_intuition": "This test file focuses on testing the user creation API endpoints."
            }}
            ```
            
            Another example with tasks:
            code: client.get('http://localhost:8000/api/tasks/{{task_id}}')
            Should extract: "/api/tasks/<task_id>"
            
            Another example with tasks:
            code: client.get('http://localhost:8000/api/tasks/{{self.__class__.task_id}}')
            Should extract: "/api/tasks/<task_id>"
            
            Another example:
            code: requests.delete('http://localhost:8000/tasks/{{task_id}}')
            Should extract: "/tasks/<task_id>"
            """)
        ])
        
        chain = prompt | llm | parser
        result = chain.invoke({
            "start_line": start_line,
            "end_line": end_line,
            "chunk_content": chunk_content,
            "import_statements": import_statements or [],
            "format_instructions": parser.get_format_instructions()
        })
        
        return {
            "test_cases": [test_case.dict() for test_case in result.test_cases],
            "chunk_intuition": result.chunk_intuition
        }
        
    except Exception as e:
        #logger.error(f"Error analyzing chunk: {str(e)}")
        return {
            "test_cases": [],
            "chunk_intuition": f"Error analyzing chunk: {str(e)}"
        }

def _generate_test_file_metadata(
    test_content: str,
    test_cases: List[Dict],
    test_intuition: str,
    llm
) -> Dict[str, Any]:
    """
    Generate simplified metadata for a test file with only 4 essential string properties.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=TestFileMetadata)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in software testing methodologies and practices across multiple programming languages.
            You must generate valid JSON output that can be parsed correctly.
            
            Based on the provided test file content and extracted test cases, generate concise metadata
            that captures the essential information about this test file in exactly 4 string properties.
            
            CRITICAL JSON FORMATTING RULES:
            - Use simple, clear sentences without special characters that could break JSON parsing
            - Avoid using backslashes, escaped characters, or complex formatting
            - Keep strings concise but informative
            - Do not use underscores in API endpoint descriptions - use simple text like "task ID" instead of "task_id"
            - Ensure all quotes are properly escaped
            - Focus on clarity and simplicity
            
            The 4 properties must cover:
            1. file_purpose: What this test file is testing (purpose and scope)
            2. test_framework_and_approach: The testing framework and overall methodology  
            3. test_coverage_summary: Summary of test coverage and what is being validated
            4. key_insights: Key patterns and important characteristics of the tests"""),
            
            ("user", """
            Test file content sample:
            ```
            {test_content_sample}
            ```
            
            Extracted test cases:
            {test_cases}
            
            Accumulated test intuition:
            "{test_intuition}"
            
            Generate metadata with exactly 4 string properties. Keep all text simple and JSON-safe.
            Avoid complex formatting, escaped characters, or technical symbols that could break JSON parsing.
            
            Example of good output format:
            {{
                "file_purpose": "Tests user authentication and session management functionality",
                "test_framework_and_approach": "Uses pytest framework with fixtures and mocking for isolated unit testing",
                "test_coverage_summary": "Covers login validation, password verification, and session timeout scenarios",
                "key_insights": "Focuses on security edge cases and error handling with comprehensive assertion patterns"
            }}
            
            {format_instructions}
            """)
        ])
        
        # Use a sample of the test content (first 1000 chars)
        test_content_sample = test_content[:1000] + "..." if len(test_content) > 1000 else test_content
        
        chain = prompt | llm | parser
        metadata = chain.invoke({
            "test_content_sample": test_content_sample,
            "test_cases": test_cases,
            "test_intuition": test_intuition,
            "format_instructions": parser.get_format_instructions()
        })
        
        return metadata.dict()
        
    except Exception as e:
        #logger.error(f"Error generating test file metadata: {str(e)}")
        return {
            "file_purpose": f"Error generating metadata: {str(e)}",
            "test_framework_and_approach": "Unknown due to error",
            "test_coverage_summary": "Unknown due to error",
            "key_insights": test_intuition or f"Error during analysis: {str(e)}"
        }
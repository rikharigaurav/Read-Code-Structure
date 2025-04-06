from typing import Dict, List, Any, Optional
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define schemas for the output structure
class CodeAnalysis(BaseModel):
    purpose: str = Field(description="One-line description of the code's purpose")
    intuition: str = Field(description="Brief explanation of how the code works in a few lines")
    properties: Dict[str, str] = Field(description="Key properties, parameters, or important aspects of the code as key-value pairs")

class FileMetadata(BaseModel):
            file_purpose: str = Field(description="Overall purpose of this file in the project")
            technologies: Dict[str, str] = Field(description="Technologies, libraries, and frameworks used in this file with brief descriptions")
            main_functionality: str = Field(description="Main functionality provided by this file")
            project_contribution: str = Field(description="How this file contributes to the overall project")
            dependencies: List[str] = Field(description="Main external dependencies this file relies on")
            complexity_assessment: str = Field(description="Assessment of the file's complexity and architecture")
            key_components: Dict[str, str] = Field(description="Key functions, classes or components with brief descriptions")

class CodeChunk(BaseModel):
    start_line: int
    end_line: int
    content: str

def analyze_code(
    code_content: str, 
    file_structure: Optional[Dict[str, Any]] = None,
    max_chunk_chars: int = 4000,  # Using character count instead of tokens
    overlap_chars: int = 400
) -> CodeAnalysis:
    """
    Analyze code content using language models, with optional chunking for large files.
    Avoids using embeddings for chunking to prevent API errors.
    """
    try:
        chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
        
        # Skip using embeddings entirely for chunking
        # Instead use character count as a proxy for complexity
        if len(code_content) < max_chunk_chars:
            return _analyze_single_chunk(code_content, file_structure, chat)
        else:
            chunks = _chunk_code_by_chars(code_content, max_chunk_chars, overlap_chars)
            return _analyze_chunked_code(chunks, file_structure, chat)
    
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        # Return a basic analysis indicating the error
        return CodeAnalysis(
            purpose="Error analyzing code",
            intuition=f"Analysis failed with error: {str(e)}",
            properties={"error": str(e)}
        )

def _chunk_code_by_chars(
    code_content: str,
    max_chars: int,
    overlap_chars: int
) -> List[CodeChunk]:
    """
    Split code into overlapping chunks based on character count
    without relying on embeddings.
    """
    lines = code_content.split('\n')
    chunks = []
    current_chunk = []
    current_chars = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_chars = len(line) + 1  # +1 for newline
        
        if current_chars + line_chars > max_chars and current_chunk:
            chunks.append(CodeChunk(
                start_line=start_line,
                end_line=i-1,
                content='\n'.join(current_chunk)
            ))
            
            # Calculate overlap for the next chunk
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
        chunks.append(CodeChunk(
            start_line=start_line,
            end_line=len(lines)-1,
            content='\n'.join(current_chunk)
        ))
    
    logger.info(f"Code split into {len(chunks)} chunks")
    return chunks

def _analyze_single_chunk(
    code_content: str,
    file_structure: Optional[Dict[str, Any]],
    llm
) -> CodeAnalysis:
    """
    Analyze a single chunk of code using the LLM.
    """
    parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert code analyst with deep understanding of software architecture, design patterns, and programming languages.
        Analyze the given code thoroughly to extract:
        1. Its core purpose (one clear sentence)
        2. An intuitive explanation of how it works (3-5 sentences)
        3. Key properties, parameters, and important aspects
        
        Focus on the big picture first, then drill down to specific details.
        If code is part of a larger structure (provided), consider that context.
        
        IMPORTANT: The "properties" field must be a dictionary with string keys and string values only.
        DO NOT use lists or arrays for any property values.
        Each property should be a key-value pair where both key and value are strings.
        
        For example, instead of:
        "functionality": ["Validation", "Database interaction"]
        
        Use this format:
        "functionality": "Validation, Database interaction"
        
        Return your analysis in the specified JSON format."""),
        ("user", """
        Code to analyze:
        ```
        {code_content}
        ```
        
        {file_structure_context}
        
        {format_instructions}
        
        Example Output:
        ```json
        {{
            "purpose": "Implements a binary search algorithm to efficiently find an element in a sorted list.",
            "intuition": "The function uses a divide-and-conquer strategy to repeatedly narrow down the search space by comparing the target with the middle element of the list. If the middle element matches the target, the search ends. Otherwise, it recursively searches in the left or right subarray depending on whether the target is smaller or larger than the middle element.",
            "properties": {{
                "time_complexity": "O(log n)",
                "space_complexity": "O(1)",
                "input": "A sorted list and a target value",
                "output": "Index of the target in the list or -1 if not found",
                "functionality": "Binary search, divide-and-conquer strategy, efficient lookup"
            }}
        }}
        ```
        
        REMEMBER: All values in the "properties" dictionary MUST be strings, not lists or arrays.
        """)
    ])
    
    # Prepare file structure context if available
    file_context = ""
    if file_structure:
        file_context = f"This code is part of a larger file structure: {file_structure}"
    
    # Run the chain
    try:
        chain = prompt | llm | parser
        return chain.invoke({
            "code_content": code_content,
            "file_structure_context": file_context,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"Error in _analyze_single_chunk: {str(e)}")
        return CodeAnalysis(
            purpose="Error analyzing code chunk",
            intuition=f"Analysis failed with error: {str(e)}",
            properties={"error": str(e)}
        )

def _analyze_chunked_code(
    chunks: List[CodeChunk],
    file_structure: Optional[Dict[str, Any]],
    llm
) -> CodeAnalysis:
    """
    Analyze code that has been split into chunks, then synthesize the results.
    """
    # First, analyze each chunk individually
    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a CHUNK of code that is part of a larger code snippet.
            Focus on understanding what this specific part does, but be aware it's incomplete.
            
            For this chunk, identify:
            1. What this specific code section seems to do
            2. How it fits into a larger codebase
            3. Key variables, functions, or patterns you observe
            
            IMPORTANT: The "properties" field must be a dictionary with string keys and string values only.
            DO NOT use lists or arrays for any property values.
            Each property should be a key-value pair where both key and value are strings.
            
            For example, instead of:
            "functionality": ["Validation", "Database interaction"]
            
            Use this format:
            "functionality": "Validation, Database interaction"
            
            Be precise but tentative, as you're only seeing part of the code."""),
            ("user", """
            Code chunk {chunk_num}/{total_chunks}:
            ```
            {code_content}
            ```
            
            Lines: {start_line} to {end_line}
            
            {file_structure_context}
            
            {format_instructions}
            
            Example Output:
            ```json
            {{
                "purpose": "Implements a binary search algorithm to efficiently find an element in a sorted list.",
                "intuition": "The function uses a divide-and-conquer strategy to repeatedly narrow down the search space by comparing the target with the middle element of the list. If the middle element matches the target, the search ends. Otherwise, it recursively searches in the left or right subarray depending on whether the target is smaller or larger than the middle element.",
                "properties": {{
                    "time_complexity": "O(log n)",
                    "space_complexity": "O(1)",
                    "input": "A sorted list and a target value",
                    "output": "Index of the target in the list or -1 if not found",
                    "functionality": "Binary search, divide-and-conquer strategy, efficient lookup"
                }}
            }}
            ```
            
            REMEMBER: All values in the "properties" dictionary MUST be strings, not lists or arrays.
            """)
        ])
        
        file_context = ""
        if file_structure:
            file_context = f"This code is part of a larger file structure: {file_structure}"
        
        try:
            chain = prompt | llm | parser
            analysis = chain.invoke({
                "chunk_num": i+1,
                "total_chunks": len(chunks),
                "code_content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "file_structure_context": file_context,
                "format_instructions": parser.get_format_instructions()
            })
            chunk_analyses.append(analysis)
        except Exception as e:
            logger.error(f"Error analyzing chunk {i+1}: {str(e)}")
            chunk_analyses.append(CodeAnalysis(
                purpose=f"Error analyzing chunk {i+1}",
                intuition=f"Analysis failed with error: {str(e)}",
                properties={"error": str(e), "chunk": f"{i+1}/{len(chunks)}"}
            ))
    
    # If all chunk analyses failed, return an error
    if all("Error" in analysis.purpose for analysis in chunk_analyses):
        return CodeAnalysis(
            purpose="Error analyzing all code chunks",
            intuition="All chunk analyses failed with errors",
            properties={"errors": str([analysis.properties.get("error", "Unknown error") for analysis in chunk_analyses])}
        )
    
    # Now synthesize the chunk analyses into a coherent whole
    synthesizer_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    
    synthesizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are synthesizing multiple analyses of code chunks into one coherent analysis.
        Your task is to create a unified understanding of the entire code snippet based on analyses of its parts.
        
        Create:
        1. A single clear purpose statement for the entire code
        2. A concise intuitive explanation of the code's overall functionality
        3. A consolidated list of important properties and parameters
        
        IMPORTANT: The "properties" field must be a dictionary with string keys and string values only.
        DO NOT use lists or arrays for any property values.
        Each property should be a key-value pair where both key and value are strings.
        
        For example, instead of:
        "functionality": ["Validation", "Database interaction"]
        
        Use this format:
        "functionality": "Validation, Database interaction"
        
        Be comprehensive but avoid redundancy. Prioritize information from all chunks.
        Resolve any contradictions between chunk analyses by considering the broader context."""),
        ("user", """
        Individual chunk analyses:
        
        {chunk_analyses}
        
        {file_structure_context}
        
        {format_instructions}
        
        Example Output:
        ```json
        {{
            "purpose": "Implements a binary search algorithm to efficiently find an element in a sorted list.",
            "intuition": "The function uses a divide-and-conquer strategy to repeatedly narrow down the search space by comparing the target with the middle element of the list. If the middle element matches the target, the search ends. Otherwise, it recursively searches in the left or right subarray depending on whether the target is smaller or larger than the middle element.",
            "properties": {{
                "time_complexity": "O(log n)",
                "space_complexity": "O(1)",
                "input": "A sorted list and a target value",
                "output": "Index of the target in the list or -1 if not found",
                "functionality": "Binary search, divide-and-conquer strategy, efficient lookup"
            }}
        }}
        ```
        
        REMEMBER: All values in the "properties" dictionary MUST be strings, not lists or arrays.
        """)
    ])
    
    file_context = ""
    if file_structure:
        file_context = f"This code is part of a larger file structure: {file_structure}"
    
    # Format the chunk analyses for the prompt
    analyses_text = "\n\n".join([
        f"CHUNK {i+1}:\n" +
        f"Purpose: {analysis.purpose}\n" +
        f"Intuition: {analysis.intuition}\n" +
        f"Properties: {analysis.properties}"
        for i, analysis in enumerate(chunk_analyses)
    ])
    
    try:
        chain = synthesizer_prompt | llm | synthesizer_parser
        return chain.invoke({
            "chunk_analyses": analyses_text,
            "file_structure_context": file_context,
            "format_instructions": synthesizer_parser.get_format_instructions()
        })
    except Exception as e:
        logger.error(f"Error synthesizing chunks: {str(e)}")
        return CodeAnalysis(
            purpose="Error synthesizing code analysis",
            intuition=f"Synthesis failed with error: {str(e)}",
            properties={"error": str(e), "individual_analyses": "Completed but synthesis failed"}
        )

# Alternative implementation if you need to use embeddings for some other purpose
def get_embeddings_with_proper_params():
    """
    Create a properly configured Cohere embeddings instance.
    Only use this if embeddings are actually needed for a specific purpose.
    """
    try:
        # Try all possible input_type values that Cohere might accept
        for input_type in ["search_document", "search_query", "classification", "clustering"]:
            try:
                embedding = CohereEmbeddings(
                    cohere_api_key=os.getenv("COHERE_API_KEY_3"),
                    model="embed-english-v3.0",
                    input_type=input_type
                )
                # Test with a simple string to verify it works
                result = embedding.embed_query("Test string")
                logger.info(f"Embeddings working with input_type={input_type}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed with input_type={input_type}: {str(e)}")
                continue
        
        # If all attempts failed, log a comprehensive error
        logger.error("All embedding attempts failed with different input_type values")
        return None
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return None 
    

def generate_file_metadata(file_path, file_structure, conversation_history, llm):
    """
    Generate comprehensive metadata for a code file based on its content and the
    conversation history accumulated during processing.
    
    Args:
        file_path: Path to the file
        file_structure: Parsed structure of the file
        code_content: Raw code content
        conversation_history: Accumulated conversation history from processing
        llm: Language model instance to use
    
    Returns:
        Dictionary containing metadata about the file
    """
    try:
        # Create a prompt for metadata generation
        parser = PydanticOutputParser(pydantic_object=BaseModel)
        
        # Extract file name and extension
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1]
        
        # Create a condensed version of the conversation history
        condensed_history = "\n".join([
            f"Function/API: {item.get('name', 'Unknown')}\n" +
            f"Purpose: {item.get('purpose', 'Unknown')}\n" +
            f"Summary: {item.get('summary', 'Unknown')}"
            for item in conversation_history
            if isinstance(item, dict) and 'name' in item
        ])
            
        parser = PydanticOutputParser(pydantic_object=FileMetadata)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code analyst tasked with generating comprehensive metadata for a code file.
            Based on the file content, structure, and processing history, generate metadata that will help in:
            1. Understanding the file's purpose in the project
            2. Identifying the technologies used
            3. Explaining how the file contributes to the project
            4. Highlighting key components and dependencies
            
            The metadata should be comprehensive yet concise and focused on information that would be valuable
            for code search, retrieval, and understanding the file's role in the project.
            
            Do not include the raw code in your response. Focus on insights and analysis.
            """),
            ("user", """
            File Path: {file_path}
            File Extension: {file_extension}
            
            File Structure Summary:
            {file_structure}
            
            Processing History (Insights from components):
            {conversation_history}
            
            Generate comprehensive metadata for this file that captures its essence,
            purpose, technologies used, and how it fits into the broader project.
            
            {format_instructions}
            """)
        ])
        
        # Run the chain
        chain = prompt | llm | parser
        metadata = chain.invoke({
            "file_path": file_path,
            "file_extension": file_extension,
            "file_structure": str(file_structure),
            "conversation_history": condensed_history,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Convert the Pydantic model to a dictionary
        metadata_dict = metadata.dict()
        
        return metadata_dict
        
    except Exception as e:
        logging.error(f"Error generating file metadata: {str(e)}")
        return {
            "file_purpose": f"Error generating metadata: {str(e)}",
            "technologies": {"error": str(e)},
            "main_functionality": "Unknown due to error",
            "project_contribution": "Unknown due to error",
            "dependencies": ["Error during metadata generation"],
            "complexity_assessment": "Unknown due to error",
            "key_components": {"error": str(e)}
        }
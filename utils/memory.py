from typing import Dict, List, Any, Optional, Union
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

# Define schemas for the output structure
class CodeAnalysis(BaseModel):
    purpose: str = Field(description="One-line description of the code's purpose")
    intuition: str = Field(description="Brief explanation of how the code works in a few lines")
    properties: Dict[str, str] = Field(description="Key properties, parameters, or important aspects of the code")

class CodeChunk(BaseModel):
    start_line: int
    end_line: int
    content: str

def analyze_code(
    code_content: str, 
    file_structure: Optional[Dict[str, Any]] = None,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    max_chunk_tokens: int = 1024,
    overlap_tokens: int = 100
) -> CodeAnalysis:
    chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokens = tokenizer.encode(code_content, add_special_tokens=False)
    
    if len(tokens) <= max_chunk_tokens:
        return _analyze_single_chunk(code_content, file_structure, chat)
    else:
        chunks = _chunk_code(code_content, tokenizer, max_chunk_tokens, overlap_tokens)
        return _analyze_chunked_code(chunks, file_structure, chat)

def _chunk_code(
    code_content: str,
    tokenizer,
    max_tokens: int,
    overlap: int
) -> List[CodeChunk]:
    """
    Split code into overlapping chunks to preserve context.
    """
    lines = code_content.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append(CodeChunk(
                start_line=start_line,
                end_line=i-1,
                content='\n'.join(current_chunk)
            ))
            
            # Calculate overlap for the next chunk
            overlap_lines = []
            overlap_tokens_count = 0
            for prev_line in reversed(current_chunk):
                prev_line_tokens = len(tokenizer.encode(prev_line, add_special_tokens=False))
                if overlap_tokens_count + prev_line_tokens <= overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_tokens_count += prev_line_tokens
                else:
                    break
            
            current_chunk = overlap_lines
            current_tokens = overlap_tokens_count
            start_line = i - len(overlap_lines)
        
        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append(CodeChunk(
            start_line=start_line,
            end_line=len(lines)-1,
            content='\n'.join(current_chunk)
        ))
    
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
        
        Return your analysis in the specified JSON format."""),
        ("user", """
        Code to analyze:
        ```
        {code_content}
        ```
        
        {file_structure_context}
        
        {format_instructions}
        """)
    ])
    
    # Prepare file structure context if available
    file_context = ""
    if file_structure:
        file_context = f"This code is part of a larger file structure: {file_structure}"
    
    # Run the chain
    chain = prompt | llm | parser
    return chain.invoke({
        "code_content": code_content,
        "file_structure_context": file_context,
        "format_instructions": parser.get_format_instructions()
    })

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
            
            Be precise but tentative, as you're only seeing part of the code."""),
            ("user", """
            Code chunk {chunk_num}/{total_chunks}:
            ```
            {code_content}
            ```
            
            Lines: {start_line} to {end_line}
            
            {file_structure_context}
            
            {format_instructions}
            """)
        ])
        
        file_context = ""
        if file_structure:
            file_context = f"This code is part of a larger file structure: {file_structure}"
        
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
    
    # Now synthesize the chunk analyses into a coherent whole
    synthesizer_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    
    synthesizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are synthesizing multiple analyses of code chunks into one coherent analysis.
        Your task is to create a unified understanding of the entire code snippet based on analyses of its parts.
        
        Create:
        1. A single clear purpose statement for the entire code
        2. A concise intuitive explanation of the code's overall functionality
        3. A consolidated list of important properties and parameters
        
        Be comprehensive but avoid redundancy. Prioritize information from all chunks.
        Resolve any contradictions between chunk analyses by considering the broader context."""),
        ("user", """
        Individual chunk analyses:
        
        {chunk_analyses}
        
        {file_structure_context}
        
        {format_instructions}
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
    
    chain = synthesizer_prompt | llm | synthesizer_parser
    return chain.invoke({
        "chunk_analyses": analyses_text,
        "file_structure_context": file_context,
        "format_instructions": synthesizer_parser.get_format_instructions()
    })

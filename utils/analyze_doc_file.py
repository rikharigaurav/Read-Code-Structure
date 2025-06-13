from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
import json
from pathlib import Path
import os

# logging.basicConfig(level=logging.INFO)
#logger = logging.get#logger(__name__)

class PineconeDocumentationMetadata(BaseModel):
    """Simplified Pinecone-compatible metadata about a documentation file."""
    doc_purpose: str = Field(description="Overall purpose of this documentation file")
    doc_type: str = Field(description="Type of documentation (tutorial, API reference, guide, etc.)")
    summary: str = Field(description="Detailed paragraph summary of the documentation content")
    key_concepts: str = Field(description="Comma-separated string of key concepts mentioned in documentation")

class DocumentationChunk(BaseModel):
    """A chunk of documentation content for processing."""
    start_line: int
    end_line: int
    content: str

def analyze_documentation_file(
    doc_content: str,
    max_chunk_chars: int = 4000,
    overlap_chars: int = 400,
    llm = None
) -> Dict[str, Any]:
    """
    Analyze documentation file content to generate simplified metadata.
    
    Args:
        doc_content: Content of the documentation file
        max_chunk_chars: Maximum characters per chunk for processing
        overlap_chars: Character overlap between chunks
        llm: Language model to use for analysis
        
    Returns:
        Dictionary containing simplified documentation metadata
    """
    try:
        doc_insights = ""
        
        if len(doc_content) < max_chunk_chars:
            # For small files, analyze in one go
            doc_insights = _analyze_doc_chunk(doc_content, 0, len(doc_content.split('\n')), llm)
        else:
            chunks = _chunk_documentation_file(doc_content, max_chunk_chars, overlap_chars)
            
            for i, chunk in enumerate(chunks):
                chunk_insight = _analyze_doc_chunk(
                    chunk.content, 
                    chunk.start_line, 
                    chunk.end_line,
                    llm
                )
                doc_insights += chunk_insight + " "

        metadata = _generate_documentation_metadata(doc_content, doc_insights, llm)
        print(f"docs metadata {metadata}")
        
        # Return the metadata directly (not nested)
        return metadata
    
    except Exception as e:
        #logger.error(f"Error analyzing documentation file: {str(e)}")
        return {
            "doc_purpose": f"Error analyzing documentation file: {str(e)}",
            "doc_type": "Unknown due to error",
            "summary": f"Analysis failed with error: {str(e)}",
            "key_concepts": ""
        }


def _chunk_documentation_file(
    doc_content: str,
    max_chars: int, 
    overlap_chars: int
) -> List[DocumentationChunk]:
    """
    Split documentation file into overlapping chunks based on character count.
    """
    lines = doc_content.split('\n')
    chunks = []
    current_chunk = []
    current_chars = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_chars = len(line) + 1 
        if current_chars + line_chars > max_chars and current_chunk:
            chunks.append(DocumentationChunk(
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
        chunks.append(DocumentationChunk(
            start_line=start_line,
            end_line=len(lines)-1,
            content='\n'.join(current_chunk)
        ))
    
    # #logger.info(f"Documentation file split into {len(chunks)} chunks")
    return chunks

def _analyze_doc_chunk(
    chunk_content: str,
    start_line: int,
    end_line: int,
    llm
) -> str:
    """
    Analyze a single chunk of documentation file to extract insights.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert documentation analyst with deep understanding of software documentation formats,
            technical writing, and code structures across multiple programming languages.
            
            You're analyzing a documentation file (or chunk of documentation) to extract key insights about
            what this documentation explains. Focus on:
            
            1. What components or features are being documented
            2. Key functionality or concepts explained
            3. Usage patterns or examples provided
            4. Common issues or edge cases mentioned
            5. How this documentation might help troubleshoot issues
            
            Provide a concise summary of the key insights from this documentation chunk."""),
            
            ("user", """Documentation file chunk (Lines {start_line} to {end_line}):
            ```
            {chunk_content}
            ```
            
            Provide key insights about what this documentation explains.
            """)
        ])
        
        chain = prompt | llm
        result = chain.invoke({
            "start_line": start_line,
            "end_line": end_line,
            "chunk_content": chunk_content
        })
        
        return result
        
    except Exception as e:
        #logger.error(f"Error analyzing documentation chunk: {str(e)}")
        return f"Error analyzing chunk: {str(e)}"

def _generate_documentation_metadata(
    doc_content: str,
    doc_insights: str,
    llm
) -> Dict[str, Any]:
    """
    Generate simplified metadata for a documentation file based on accumulated insights.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=PineconeDocumentationMetadata)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in technical documentation analysis and software architecture.
            Your task is to generate simplified metadata for a documentation file that will help developers
            understand its relevance to their work.
            
            You must return EXACTLY 4 properties in JSON format with this exact structure:
            
            EXAMPLE OUTPUT FORMAT:
            {{
                "doc_purpose": "Provides installation and setup instructions for the Flask web framework",
                "doc_type": "Installation Guide",
                "summary": "This comprehensive installation guide walks developers through the complete process of setting up Flask in their development environment. It covers multiple installation methods including pip installation, virtual environment creation and activation, dependency management best practices, and initial project configuration. The documentation addresses common installation pitfalls, provides troubleshooting steps for typical dependency conflicts, explains how to verify successful installation, and includes platform-specific considerations for Windows, macOS, and Linux systems. Additionally, it demonstrates basic Flask application structure and includes examples of minimal working applications to help developers validate their setup.",
                "key_concepts": "Flask installation, virtual environments, pip package manager, dependency management, configuration setup, troubleshooting, cross-platform compatibility"
            }}
            
            PROPERTY DEFINITIONS:
            1. doc_purpose: The overall purpose/goal of this documentation (string)
            2. doc_type: Type of documentation - choose from: "README", "API Reference", "Tutorial", "Installation Guide", "User Guide", "Developer Guide", "Configuration Guide", "Troubleshooting Guide", "Release Notes", "Contributing Guide", or "Technical Specification" (string)
            3. summary: A detailed paragraph (4-6 sentences) that thoroughly explains what the documentation covers, what developers will learn, key topics addressed, and how it helps solve problems or accomplish tasks (string)
            4. key_concepts: A comma-separated string of 5-10 important technical concepts, features, or topics mentioned (string)
            
            Be specific and actionable. Focus on what developers would find most useful for understanding and using this documentation."""),
            
            ("user", """Documentation content sample:
            ```
            {doc_content_sample}
            ```
            
            Accumulated insights from analysis:
            "{doc_insights}"
            
            Analyze the documentation and return metadata in the exact JSON format shown in the example above.
            Ensure your response contains EXACTLY these 4 properties:
            - doc_purpose (string)
            - doc_type (string) 
            - summary (string) - must be a detailed paragraph of 4-6 sentences
            - key_concepts (string) - comma-separated values
            
            {format_instructions}
            """)
        ])
        
        if len(doc_content) > 2000:
            doc_content_sample = doc_content[:1500] + "\n...\n" + doc_content[-500:]
        else:
            doc_content_sample = doc_content
        
        chain = prompt | llm | parser
        metadata = chain.invoke({
            "doc_content_sample": doc_content_sample,
            "doc_insights": doc_insights,
            "format_instructions": parser.get_format_instructions()
        })
        
        return metadata.dict()
        
    except Exception as e:
        #logger.error(f"Error generating documentation metadata: {str(e)}")
        return {
            "doc_purpose": f"Error generating metadata: {str(e)}",
            "doc_type": "Unknown due to error",
            "summary": doc_insights or f"Error during analysis: {str(e)}",
            "key_concepts": ""
        }
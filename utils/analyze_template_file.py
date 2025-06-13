from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# logging.basicConfig(level=logging.INFO)
#logger = logging.get#logger(__name__)

class MarkupTemplateMetadata(BaseModel):
    """Metadata about a markup template file."""
    template_purpose: str = Field(description="Overall purpose of this markup template")
    template_type: str = Field(description="Type of markup template (HTML, XML, Markdown, etc.)")
    target_platforms: List[str] = Field(description="Intended platforms or applications")
    markup_components: List[str] = Field(description="Main markup components used in template")
    style_references: List[str] = Field(description="Style references or frameworks mentioned")
    summary: str = Field(description="Concise summary of the template content")
    common_usages: str = Field(description="How this template might typically be used")
    key_elements: List[str] = Field(description="List of key elements or tags in the template")
    element_descriptions: List[str] = Field(description="Descriptions of key elements, in same order as key_elements")

class MarkupChunk(BaseModel):
    """A chunk of markup content for processing."""
    start_line: int
    end_line: int
    content: str

def analyze_markup_template(
    template_content: str,
    max_chunk_chars: int = 4000,
    overlap_chars: int = 400,
    llm = None
) -> Dict[str, Any]:
    """
    Analyze markup template content to generate metadata about its purpose and structure.
    
    Args:
        template_content: Content of the markup template file
        max_chunk_chars: Maximum characters per chunk for processing
        overlap_chars: Character overlap between chunks
        llm: Language model to use for analysis
        
    Returns:
        Dictionary containing template metadata
    """
    try:
        template_insights = ""
        
        if len(template_content) < max_chunk_chars:
            # For small files, analyze in one go
            template_insights = _analyze_markup_chunk(template_content, 0, len(template_content.split('\n')), llm)
        else:
            chunks = _chunk_markup_file(template_content, max_chunk_chars, overlap_chars)
            
            for i, chunk in enumerate(chunks):
                chunk_insight = _analyze_markup_chunk(
                    chunk.content, 
                    chunk.start_line, 
                    chunk.end_line,
                    llm
                )
                template_insights += chunk_insight + " "

        metadata = _generate_template_metadata(template_content, template_insights, llm)
        
        return {
            "metadata": metadata
        }
    
    except Exception as e:
        #logger.error(f"Error analyzing markup template: {str(e)}")
        return {
            "metadata": {
                "template_purpose": f"Error analyzing markup template: {str(e)}",
                "template_type": "Unknown due to error",
                "target_platforms": ["Unknown due to error"],
                "markup_components": [],
                "style_references": [],
                "summary": f"Analysis failed with error: {str(e)}",
                "common_usages": "Unable to determine due to error",
                "key_elements": [],
                "element_descriptions": []
            }
        }

def _chunk_markup_file(
    template_content: str,
    max_chars: int, 
    overlap_chars: int
) -> List[MarkupChunk]:
    """
    Split markup template file into overlapping chunks based on character count.
    """
    lines = template_content.split('\n')
    chunks = []
    current_chunk = []
    current_chars = 0
    start_line = 0
    
    for i, line in enumerate(lines):
        line_chars = len(line) + 1 
        if current_chars + line_chars > max_chars and current_chunk:
            chunks.append(MarkupChunk(
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
        chunks.append(MarkupChunk(
            start_line=start_line,
            end_line=len(lines)-1,
            content='\n'.join(current_chunk)
        ))
    
    #logger.info(f"Markup template split into {len(chunks)} chunks")
    return chunks

def _analyze_markup_chunk(
    chunk_content: str,
    start_line: int,
    end_line: int,
    llm
) -> str:
    """
    Analyze a single chunk of markup template to extract insights.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert markup and template analyst with deep understanding of HTML, XML, 
            Markdown, and other markup languages, as well as templating systems like Jinja, Mustache, etc.
            
            You're analyzing a markup template file (or chunk of template) to extract key insights about
            what this template contains and how it's structured. Focus on:
            
            1. What components or elements are being used
            2. Structure and organization of the markup
            3. Template variables or placeholders used
            4. Style references or visual components
            5. How this template might be implemented in real applications
            
            Provide a concise summary of the key insights from this template chunk."""),
            
            ("user", """Markup template chunk (Lines {start_line} to {end_line}):
            ```
            {chunk_content}
            ```
            
            Provide key insights about what this markup template contains and how it's structured.
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
        #logger.error(f"Error analyzing markup chunk: {str(e)}")
        return f"Error analyzing chunk: {str(e)}"

def _generate_template_metadata(
    template_content: str,
    template_insights: str,
    llm
) -> Dict[str, Any]:
    """
    Generate metadata for a markup template file based on accumulated insights.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=MarkupTemplateMetadata)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in markup languages, web development, and template analysis.
            Your task is to generate comprehensive metadata for a markup template file that will help developers
            understand its structure, purpose, and how to use it effectively.
            
            Focus on extracting:
            1. The overall purpose of this template
            2. The type of markup (HTML, XML, Markdown, etc.)
            3. The intended platforms or applications (as a list of strings)
            4. Key markup components used (as a list of strings)
            5. Important elements or tags (as a list of strings)
            6. Descriptions of those key elements (as a parallel list of strings in the same order)
            7. Style references or frameworks mentioned (as a list of strings)
            8. A concise summary of the template content
            9. How this template might typically be used
            
            Be specific and actionable in your analysis. The goal is to help developers quickly understand
            what this template contains and how they might adapt it for their own projects."""),
            
            ("user", """Template content sample:
            ```
            {template_content_sample}
            ```
            
            Accumulated insights from analysis:
            "{template_insights}"
            
            Generate comprehensive metadata for this markup template that is compatible with a structured database
            that only accepts strings, numbers, and lists of strings (no dictionaries or complex types).
            
            For 'key elements', provide two parallel lists:
            1. 'key_elements': A list of element or tag names
            2. 'element_descriptions': A list of descriptions in the same order as the elements
            
            {format_instructions}
            """)
        ])
        if len(template_content) > 2000:
            template_content_sample = template_content[:1500] + "\n...\n" + template_content[-500:]
        else:
            template_content_sample = template_content
        
        chain = prompt | llm | parser
        metadata = chain.invoke({
            "template_content_sample": template_content_sample,
            "template_insights": template_insights,
            "format_instructions": parser.get_format_instructions()
        })
        
        return metadata.dict()
        
    except Exception as e:
        #logger.error(f"Error generating template metadata: {str(e)}")
        return {
            "template_purpose": f"Error generating metadata: {str(e)}",
            "template_type": "Unknown due to error",
            "target_platforms": ["Unknown due to error"],
            "markup_components": [],
            "style_references": [],
            "summary": template_insights or f"Error during analysis: {str(e)}",
            "common_usages": "Unable to determine due to error",
            "key_elements": [],
            "element_descriptions": []
        }
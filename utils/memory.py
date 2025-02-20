from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.sqlite import SqliteSaver
from transformers import AutoTokenizer
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from utils.treeSitter import process_file

chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
# Define state structure

# class GraphFile(BaseModel):
#     summary: str = Field(description = "Summary of the current node")
#     file_type: str = Field(description="Format of the file (e.g., DOT, GraphML, JSON, YAML).")

class SourceCodeFile(BaseModel):
    summary: str = Field(description = "Summary of the current node")

class CLASS(BaseModel): 
    functionCalls: list = Field(description=" contains all the function calls inside this class ")
    apiCalls: list = Field(description="contains all the API ROUTE calls inside this class")

class Function(BaseModel):
    arguments: list[dict] = Field(description="list all the arg name and their type")
    functionCalls: list = Field(description=" contains all the function calls inside this class ")
    apiCalls: list = Field(description="contains all the API ROUTE calls inside this class")
    returnType: str = Field(description="Return type of this function")

class ApiEndpoint(BaseModel):
    routeURL: str = Field(description="Route Url of this endpoint")
    method: str = Field(description="Http method of this endpoint")
    functionCalls: list = Field(description=" contains all the function calls inside this class ")
    apiCalls: list = Field(description="contains all the API ROUTE calls inside this class")

class TemplateMarkupFile(BaseModel):
    summary: str = Field(description = "Summary of the current node")
    file_type: str = Field(description="Template engine or markup language, such as HTML, Handlebars, EJS, or Pug")
    dependencies: list[str] = Field(description="List of files or assets (images, stylesheets, scripts) linked or imported by the template")

class TestingFile(BaseModel):
    summary: str = Field(description = "Summary of the current node")
    test_framework: list = Field(description="The testing framework used, such as Jest, Mocha, JUnit, or Pytest")
    test_reference_dict: dict = Field(
        description="Dictionary containing key-value pairs of test type and reference function or class name"
    )

class DocumentationFile(BaseModel):
    summary: str = Field(description = "Summary of the current node")
    file_type: str = Field(description="Format of the documentation, such as Markdown or reStructuredText")
    purpose: str = Field(description="The goal of the documentation, such as API documentation, user guide, or design document")

class ProcessingState(TypedDict):
    file_path: str
    content: str
    fragments: List[str]
    current_index: int
    total_fragments: int
    results: list[str]
    memory: ConversationBufferMemory
    prompt: str
    fileSchema: Any
    final_summary: dict

# Define nodes
def initialize_fragments(state: ProcessingState) -> dict:
    """Node 1: Split content into fragments"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(state["content"])
    max_tokens = 4000
    fragments = []
    
    for i in range(0, len(tokens), max_tokens):
        fragment_tokens = tokens[i:i+max_tokens]
        fragment = tokenizer.decode(fragment_tokens)
        if i + max_tokens < len(tokens) and '.' in fragment:
            fragment = fragment.rsplit('.', 1)[0] + '.'
        fragments.append(fragment)
    
    return {
        "fragments": fragments,
        "current_index": 0,
        "results": [],
        "total_fragments": len(fragments)
    }

def process_fragment_node(state: ProcessingState) -> dict:
    """Node 3: Process individual fragment with memory"""
    current_index = state["current_index"]
    fragment = state["fragments"][current_index]
    total_fragments = state["total_fragments"]
    prompt = state["prompt"]
    memory = state["memory"]
    fileSchema = state["fileSchema"]

    # Construct input message
    document_status = "[START]" if current_index == 0 else "[CONTINUED]"
    document_end = "[END]" if current_index == total_fragments - 1 else "[MORE]"

    # parser = JsonOutputParser(pydantic_object = )

    prompt = PromptTemplate(
        template='''
        Fragment {current_index_plus_1}/{total_fragments} {document_status}
        Prompt: {query}
        Content:
        {fragment}
        {document_end}
        
        Instructions:
        - Maintain context between fragments
        - Prepare for potential continuation
        - Delay final conclusions until last fragment
        ''',
        input_variables = [
            'current_index_plus_1',
            'total_fragments',
            'document_status',
            'fragment',
            'document_end',
            'query'
        ]
        # partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | chat  # | parser

# Usage example
    response = chain.invoke({
        'current_index_plus_1': current_index + 1,
        'total_fragments': total_fragments,
        'document_status': document_status,
        'fragment': fragment,
        'document_end': document_end,
        'query': prompt
    })
    
    # Update memory
    memory.save_context({"input": prompt}, {"output": response.json()})
    
    # Store result
    new_results = state["results"] + [response]

    return {
        "results": new_results,
        "current_index": current_index + 1,
        "memory": memory
    }

def generate_summary(state: ProcessingState) -> dict:
    """Node 3: Generate final summary using accumulated memory"""
    # Get conversation history from memory
    history = state["memory"].load_memory_variables({})
    chat_history = "\n".join(
        f"Input: {entry['input']}\nOutput: {entry['output']}"
        for entry in history['history']
    )
    parser = JsonOutputParser(pydantic_object=state['fileSchema'])

    import_statement = process_file(state['file_path'])
    
    summary_prompt = PromptTemplate(
        template='''
        Analyze and synthesize fragments into structured summary:
        
        File Type Context: {file_type_hint}
        Analysis Fragments:
        {chat_history}

        The import statement for the file are
        {import_statement}
        
        Create JSON summary with fields matching the file type:
        {format_instructions}
        
        Examples:
        - Graph File:
            {{"file_type": "graph", "summary": "...", "format": "DOT"}}
            
            - Test File: 
            {{"file_type": "test", "summary": "...", 
                "test_framework": ["Jest"], 
                "test_reference_dict": {{"unit": ["UserService"]}}}}
                
            - Template File:
            {{"file_type": "template", "summary": "...", 
                "template_engine": "Handlebars",
                "dependencies": ["styles.css", "header.partial"]}}
            
        Validation Rules:
        - Include ONLY fields relevant to the file type
        - Maintain strict typing for each field
        - Omit unused optional fields
        ''',
        input_variables=["chat_history", "file_type_hint"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    # Get file type hint from processing state
    file_type_hint = state['fileSchema']
    
    chain = summary_prompt | chat | parser
    summary = chain.invoke({
        "chat_history": chat_history,
        "file_type_hint": file_type_hint,
        "import_statement": import_statement
    })
    
    return {"final_summary": summary.dict()}

# Conditional edge logic
def should_continue(state: ProcessingState) -> str:
    """Determine if processing should continue"""
    return "process" if state["current_index"] < state["total_fragments"] else END

# Build the graph
builder = StateGraph(ProcessingState)

# Add nodes
builder.add_node("initialize", initialize_fragments)
builder.add_node("process", process_fragment_node)
builder.add_node("summarize", generate_summary)  # New summary node

# Set up edges
builder.set_entry_point("initialize")
builder.add_edge("initialize", "process")
builder.add_conditional_edges(
    "process",
    should_continue,
    {"process": "process", "summarize": "summarize"}
)
builder.add_edge("summarize", END)  # End after summary

# Compile the graph (keep this unchanged)
processing_graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string(":memory:")  
)

# Updated file processing function
def process_llm_calls(file_path: str, prompt: str, fileSchema: Any) -> List[dict]:
    """Process file using LangGraph with memory"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Initialize processing
    initial_state = {
        "file_path": file_path,
        "content": content,
        "prompt": prompt,
        "fileSchema": fileSchema,
        "memory": ConversationBufferMemory(),
        "fragments": [],
        "current_index": 0,
        "results": [],
        "total_fragments": 0
    }

    # Execute the graph
    final_state = processing_graph.invoke(initial_state)
    
    # Display results
    for idx, result in enumerate(final_state["results"]):
        print(f"\nFragment {idx+1} Summary:")
        # print(result["summary"])
        # print("-" * 40)

    output = {
        "result" : final_state['result'],
        "prop" : final_state.get('final_summary')
    }
    
    return output
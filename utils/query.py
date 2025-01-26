from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError, model_validator
from langgraph.graph.message import add_messages
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate

class LangGraphState(TypedDict):
    original_query: str 
    subqueries: list[str]
    decomposed_queries: list[dict]
    aggregated_data: list[dict]
    final_result: str


graph_builder = StateGraph(LangGraphState)

llm = ChatMistralAI(os.getenv('MISTRAL_API_KEY'))

workflow = StateGraph(LangGraphState)

class subqueries(BaseModel):
    subqueries: list[str] = Field(description = "sub-query derived from the main query")

class QueryDecomposition(BaseModel):
    subquery: str = Field(
        description="The specific aspect of the original query to investigate",
        example="Find all functions that call the payment gateway API"
    )
    intent_class: Literal['semantic', 'structural', 'hybrid'] = Field(
        description="Nature of the query: semantic (meaning), structural (relationships), or both",
        example="structural"
    )
    scope: dict = Field(
        description="Target focus with type and specifics",
        example={
            "type": "class|function|endpoint|file",
            "name": "PaymentService",
            "pattern": "api/gateways/*.py"
        }
    )
    operation: Literal['similarity_search', 'dependency_tracing', 'pattern_matching', 'path_finding'] = Field(
        description="Specific operation needed to resolve the subquery",
        example="dependency_tracing"
    )
    search_type: Literal['vector', 'graph', 'hybrid'] = Field(
        description="Determined search mechanism based on other properties",
        example="graph"
    )
    context_requirements: list[str] = Field(
        description="Additional context needed for execution",
        example=["API response formats", "Error handling patterns"]
    )

class Decomposer(BaseModel):
    Query_Intent: list[QueryDecomposition]

    @model_validator(mode="before")
    @classmethod
    def validate_posts(cls, values: dict) -> dict:
        posts = values.get("posts", [])
        if not posts:
            raise ValueError("The list of posts cannot be empty.")
        if len(posts) > 100:
            raise ValueError("Too many posts. The maximum allowed is 100.")
        return values   


def create_subqueries(state):
    """
    Classify the intent of the query based on predefined categories.
    """
    original_query = state["original_query"]

    prompt = '''
        You are an expert assistant designed to analyze and process queries about large software codebases. Your task is to break down a given query into smaller, actionable sub-queries, each focusing on a specific task or aspect of the codebase. Ensure the sub-queries are:

    1. **Concise and Focused**: Each sub-query should address a single, well-defined task.
    2. **Hierarchical**: Sub-queries should represent logical steps needed to solve the main query.
    3. **Relevant**: Sub-queries should target important areas like file relationships, dependencies, semantic analysis, or bug detection.

    For example:
    Query: "Find all the circular dependencies in the repository and suggest how to fix them."
    Sub-queries:
    1. Identify all dependencies in the repository.
    2. Detect circular dependencies among them.
    3. Analyze the circular dependencies and suggest ways to resolve them.

    Now, process the following query and generate sub-queries:

    Query: {query}

    {format_instructions}
    '''

    parser = JsonOutputParser(pydantic_object = subqueries)
    
    prompt_and_parser = PromptTemplate(
        template = prompt,
        input_variables = ['query'],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
    )
    
    chain = prompt_and_parser | llm | parser

    output = chain.invoke({"query": original_query})
    print(output)
    state["subqueries"] = output.dict()
    print(output.dict())
    
    return {"subqueries": output.dict()}


def decompose_query(state):
    """
    Break down the normalized query into manageable subtasks for further processing.
    """

    subqueries = state['subqueries']

    prompt = '''
        You are an expert assistant for analyzing codebases. Your task is to decompose a given sub-query into structured parts.  
        This decomposition should identify the intent, target, and required operations to process the query effectively.  

        Use the following structure to guide your output:  
        1. **Query Intent**: The purpose or goal of the query (e.g., find issues, suggest solutions, analyze dependencies).  
        2. **Target Scope**: The specific part of the codebase the query focuses on (e.g., specific files, folders, modules, or the entire repository).  
        3. **Operational Type**: The type of operation needed to address the query (e.g., analysis, resolution, dependency mapping).

        Here are the sub-query:  
        "{sub_query}"  

        Return the response as a list of JSON object for all the sub queries using the following schema:  
        ```python
        {
            "Query_Intent": "<Purpose of the sub-query>",
            "Target_Scope": "<Specific area of the codebase>",
            "Operational_Type": "<Type of operation needed>"
        }
        {format_instructions}
    '''

    parser = JsonOutputParser(pydantic_object = Decomposer)
    vectorSearch = []
    graphSearch = []

    prompt_and_parser = PromptTemplate(
        template = prompt,
        input_variables = ['sub_query'],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
    )

    chain = prompt_and_parser | llm | parser

    output = chain.invoke({"sub_query": subqueries})

    for prop in output:
        pass


    return {"subqueries": subqueries}

def vectorCalls(state):
    decomposer = state['decomposed_queries']

    for DQ in decompose_query:
        if(DQ['search_type'] == 'vector' or DQ['search_type'] == 'hybrid'):
            
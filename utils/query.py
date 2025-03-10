from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError, model_validator
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from utils.pinecone_db import pinecone
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

class LangGraphState(TypedDict):
    original_query: str 
    subqueries: List[str]
    decomposed_queries: List[Dict[str, Any]]
    aggregated_data: List[Dict[str, Any]]
    final_result: Optional[str]

class FinalResult(BaseModel):
    knowledge: str = Field(description="the knowledge for the given query")
    insights: str = Field(description="the insights for the given query")
    code: str = Field(description="the code for the given query")


graph_builder = StateGraph(LangGraphState)
llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
workflow = StateGraph(LangGraphState)
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username="neo4j", password=os.getenv("NEO4J_PASSWORD"))

class subqueries(BaseModel):
    subqueries: list[str] = Field(description = "sub-query derived from the main query")

class QueryDecomposition(BaseModel):
    Query_Intent: str = Field(
        description="The specific aspect of the original query to investigate",
        example="Find all functions that call the payment gateway API"
    )
    Target_Scope: dict[str, str] = Field(
        description="Target focus with type and specifics",
        example={
            "type": "class|function|endpoint|file",
            "name": "PaymentService",
            "pattern": "api/gateways/*.py"
        }
    )
    Operational_Type: Literal['vector', 'graph', 'hybrid'] = Field(
        description="Determined search mechanism based on other properties",
        example="graph"
    )
    context_requirements: list[str] = Field(
        description="Additional context needed for execution",
        example=["API response formats", "Error handling patterns"]
    )

class Decomposer(BaseModel):
    list_subquery: list[QueryDecomposition]

    @model_validator(mode="before")
    @classmethod
    def validate_posts(cls, values: dict) -> dict:
        posts = values.get("posts", [])
        if not posts:
            raise ValueError("The list of posts cannot be empty.")
        if len(posts) > 100:
            raise ValueError("Too many posts. The maximum allowed is 100.")
        return values   
    
class context_output_structure(BaseModel):
    sub_query_output : str = Field(description="The final output for the selective sub query")

    @model_validator(mode="before")
    @classmethod
    def validate_output(cls, values: Dict) -> Dict:
        output = values.get("sub_query_output", "")
        if len(output) > 500:
            raise ValueError("Output exceeds 500 character limit")
        if output.count(".") > 4:
            raise ValueError("Maximum 4 sentences allowed")
        return values


def create_subqueries(state):
    """
    Classify the intent of the query based on predefined categories.
    """

    prompt = '''
        You are an expert assistant designed to analyze and process queries about large software codebases. Your task is to break down a given query into smaller, actionable sub-queries, each focusing on a specific task or aspect of the codebase. Ensure the sub-queries are:

    1. **Detailed and Focused**: Each sub-query should address a single, well-defined task but still should be pretty explanatory, should convey its purpose clearly. 
    2. **Hierarchical**: Sub-queries should represent logical steps needed to solve the main query.
    3. **Relevant**: Sub-queries should target important areas like file relationships, dependencies, semantic analysis, or bug detection.
    4. **Atmomicity**: Each sub-query should be self-contained and not dependent on other sub-queries.
    5. **Actionable**: Each sub-query should be actionable and lead to a specific result or insight.

    For example:
    Query: "Find all the circular dependencies in the repository and suggest how to fix them."
    Sub-queries:
    1. Identify all dependencies in the repository.
    2. Detect circular dependencies among them.
    3. Analyze the circular dependencies and suggest ways to resolve them.

    Now, process the following query and generate sub-queries:

    Query: {query}

    YOUR RESPONSE MUST BE VALID JSON in the following format only, with no additional text:
                {{
                    "subqueries": "sub-query derived from the main query"
                }}

    {format_instructions}
    '''

    parser = JsonOutputParser(pydantic_object = subqueries)
    
    prompt_and_parser = PromptTemplate(
        template = prompt,
        input_variables = ['query'],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
    )
    
    chain = prompt_and_parser | llm | parser

    result = chain.invoke({"query": state["original_query"]})
    print(result)
    return {"subqueries": result['subqueries']}


def decompose_query(state):
    """
    Break down the normalized query into manageable subtasks for further processing.
    """

    prompt = '''
        You are an expert assistant for analyzing codebases. Your task is to decompose a given sub-query into structured parts that will be used for precise database retrieval.
        
        IMPORTANT: The Query_Intent field will be used DIRECTLY for database retrieval, so it must be specific, concise, and contain all necessary search terms.
        
        For each sub-query, provide:
        
        1. **Query_Intent**: Write a precise, searchable description that includes:
            - All relevant technical terms and keywords
            - Specific naming patterns (if mentioned)
            - Clear indication of what needs to be found
            - The query intent should be capable enough to retrieve the required information from the database on its own so it must contains all important keywords.
            - Example: "Find all HTTP error handling patterns in API endpoint controllers" instead of just "Look for error handling"
        
        2. **Target_Scope**: Define the specific part of the codebase to focus on:
            - Be as specific as possible about file types, directories, or module names
            - Include any mentioned patterns or naming conventions
            - Specify class/function types where relevant
            - The target_scope should be in context to Query_Intent
        
        3. **Operational_Type**: Determine which search mechanism is most appropriate:
            - "vector" for semantic similarity and conceptual searches
            - "graph" for relationship-based queries and dependency analysis
            - "hybrid" for complex queries requiring both approaches
            - **Additional Guideline:** Use "graph" if the sub-query is related to structural aspects of the codebase (e.g., file architecture, class hierarchies, dependency relationships) and "vector" if the sub-query involves retrieving conceptual or knowledge-based data from the codebase.
    

        4. **context_requirements**: List any additional context needed:
            - Related code patterns
            - Documentation references
            - Configuration settings

        Here is the sub-query to decompose:
        "{sub_query}"

        Return the response as a list of JSON objects using the following schema:
        ```python
        {{
            "Query_Intent": "<Precise, database-ready search description>",
            "Target_Scope": {{
                "type": "<class|function|endpoint|file>",
                "name": "<specific name or pattern>",
                "pattern": "<file pattern or location>"
            }},
            "Operational_Type": "<vector|graph|hybrid>",
            "context_requirements": ["<requirement1>", "<requirement2>"]
        }}
        {format_instructions}
    '''

    parser = JsonOutputParser(pydantic_object = Decomposer)
    

    prompt_and_parser = PromptTemplate(
        template = prompt,
        input_variables = ['sub_query'],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
    )

    chain = prompt_and_parser | llm | parser
    result = chain.invoke({"sub_query": state["subqueries"]})
    print(f"THE DECOMPOSED QUERIES ARE ::: {result}")
    
    return {"decomposed_queries": result}

def vectorCalls(state):
    """
    Process all queries that need vector search and add results to aggregated_data.
    """
    results = []
    for query in state["decomposed_queries"]:
        # Only process queries with Operational_Type 'vector' or 'hybrid'
        if query.get('Operational_Type') in ['vector', 'hybrid']:
            # Use the Query_Intent directly for retrieval
            print(f"THE QUERY INTENT IS ::: {query.get('Query_Intent')}")
            retrieved_data = pinecone.retrieve_data_from_pincone(query.get('Query_Intent'))
            print(f"THE RETRIEVED DATA IS ::: {retrieved_data}")
            
            parser = JsonOutputParser(pydantic_object=context_output_structure)
            prompt_template = PromptTemplate(
                template = """
                Analyze the following code context to answer the query.
                Query: {subquery}
                target_scope: {Target_Scope}
                context_requirements: {context_requirements}
                Context: {retrieved_data}
                Provide a concise response that directly addresses the query based on the given context and if context is not give provide an ans based on your knowledge and the target_scope and context_requirement.
                NOTE :- DON'T PROVIDE CODE AS OUTPUT. IT SHOULD BE PROPER ENGLISH LANGUAGE SENTENCE.
                 YOUR RESPONSE MUST BE VALID JSON in the following format only, with no additional text:
                {{
                    "sub_query_output": "Your detailed answer here"
                }}
                {format_instructions}
                """,
                input_variables = ['subquery', 'Target_Scope', 'context_requirements', 'retrieved_data'],
                partial_variables = {"format_instructions": parser.get_format_instructions()}
            )
            
            prompt_model = prompt_template | llm 
            output = prompt_model.invoke({
                "subquery": query.get('Query_Intent'),
                "Target_Scope": query.get('Target_Scope'),
                "context_requirements": query.get('context_requirements'),
                "retrieved_data": retrieved_data
            })
            result = parser.invoke(output)
            print(f'THE OUTPUT FOR VECTOR RETRIEVAL IS ::: {result}')
            
            results.append({
                "subquery": query.get('Query_Intent'),
                "source": "vector",
                "result": "No result received from LLM" if result is None else result['sub_query_output'],
                "target_scope": query.get('Target_Scope'),
                "context_requirements": query.get('context_requirements')
            })
    
    return {"aggregated_data": results}
            
def graphCalls(state):
    """
    Process queries with graph search based on their Operational_Type.
    Takes the state containing decomposed_queries in the expected format.
    Includes LLM processing similar to vector calls.
    """ 
    results = state.get("aggregated_data", [])  # Start with existing results from vector search
    
    # Access decomposed_queries directly from state
    for query in state["decomposed_queries"]:
        # Only process queries with Operational_Type 'graph' or 'hybrid'
        if query.get('Operational_Type') in ['graph', 'hybrid']:
            CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a code graph database.
            Instructions:
            - Use only the provided node types, relationships, and properties in the schema.
            - Do not use any node types, relationships, or properties that are not provided in the schema.
            - Return only the Cypher query with no additional explanation.

            Schema:
            Node Types and Properties:
            1. FOLDER: {fullPath}
            - Properties: None specified

            2. SOURCECODEFILE: {file_path}:{file_extension}
            - Properties: file_id, file_name, file_path, file_extension

            3. TESTINGFILE: {fullPath}:{file_extension}
            - Properties: file_id, file_name, fullPath, file_extension, result['test_framework'], test_reference_str
            - Note: test_reference_str is a dictionary containing functions being tested by this test file

            4. DOCUMENTATIONFILE: {file_path}:{file_extension}
            - Properties: file_id, file_name, file_path, file_extension

            5. TEMPLATEMARKUPFILE: {file_path}:{file_extension}
            - Properties: file_id, file_name, file_path, file_extension

            6. APIENDPOINT: {route}:{method_name.upper()}
            - Properties: node_id, route, http_method

            7. FUNCTION: {relative_file_path}:{func_name}
            - Properties: node_id, func_name, file_path, return_type

            Relationship Types:
            1. BELONGS_TO - indicates which node belongs to (comes under) which node
            2. HTTP_METHOD - connects between API calls
            3. CALLS - used for function calling relationships

            Note: Do not include any explanations or apologies in your responses.
            Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
            Do not include any text except the generated Cypher statement.

            Examples:
            # Find all Python source code files
            MATCH (s:SOURCECODEFILE)
            WHERE s.file_extension = 'py'
            RETURN s.file_name, s.file_path

            # Find functions that call other functions
            MATCH (f1:FUNCTION)-[:CALLS]->(f2:FUNCTION)
            RETURN f1.func_name AS caller, f2.func_name AS callee

            # Find test files that test a specific function
            MATCH (t:TESTINGFILE)-[:BELONGS_TO]->(f:FUNCTION)
            WHERE f.func_name = 'process_data'
            RETURN t.file_name

            # Find all API endpoints and their HTTP methods
            MATCH (a:APIENDPOINT)
            RETURN a.route, a.http_method

            # Find source code files in a specific folder
            MATCH (f:FOLDER)<-[:BELONGS_TO]-(s:SOURCECODEFILE)
            WHERE f.fullPath CONTAINS '/src/models'
            RETURN s.file_name, s.file_path

            The question is:
            {question}"""

            CYPHER_GENERATION_PROMPT = PromptTemplate(
                input_variables=["question"], 
                template=CYPHER_GENERATION_TEMPLATE
            )
            chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=True,
                validate_cypher=False,
                allow_dangerous_requests=True,
                cypher_prompt=CYPHER_GENERATION_PROMPT
            )
            
            # Use the Query_Intent directly for the graph query
            graph_response = chain.invoke({"query": query.get('Query_Intent')})
            print(f"THE GRAPH RESPONSE IS ::: {graph_response}")
            # Process the graph results with LLM (similar to vector call)
            parser = JsonOutputParser(pydantic_object=context_output_structure)
            prompt_template = PromptTemplate(
                template = """
                Analyze the following code context to answer the query.
                Query: {subquery}
                target_scope: {Target_Scope}
                context_requirements: {context_requirements}
                Context: {retrieved_data}
                Provide a concise response that directly addresses the query based on the given context and if context is not give provide an ans based on your knowledge and the target_scope and context_requirement.
                NOTE :- DON'T PROVIDE CODE AS OUTPUT. IT SHOULD BE PROPER ENGLISH LANGUAGE SENTENCE.
                 YOUR RESPONSE MUST BE VALID JSON in the following format only, with no additional text:
                {{
                    "sub_query_output": "Your detailed answer here"
                }}
                """,
                input_variables = ['subquery', 'Target_Scope', 'context_requirements', 'retrieved_data'],
                partial_variables = {"format_instruction": parser.get_format_instruction()}
            )
            
            prompt_model = prompt_template | llm | parser
            output = prompt_model.invoke({
                "subquery": query.get('Query_Intent'),
                "Target_Scope": query.get('Target_Scope'),
                "context_requirements": query.get('context_requirements'),
                "graph_results": graph_response['result']
            })
            print(f'THE OUTPUT FOR GRAPH RETRIEVAL IS ::: {output}')
            # result = output.sub_query_output if hasattr(output, 'sub_query_output') else output.get('sub_query_output', "No result found")
            results.append({
                "subquery": query.get('Query_Intent'),
                "source": "graph",
                "result": "No result received from LLM" if output is None else output['sub_query_output'],
                "target_scope": query.get('Target_Scope'),
                "context_requirements": query.get('context_requirements')
            })
    
    return {"aggregated_data": results}

def aggregateResults(state):
    """
    Aggregate all results from vector and graph searches into a final cohesive response.
    """
    aggregated_data = state.get("aggregated_data", [])
    
    if not aggregated_data:
        return {"final_result": "No results were found for your query."}
    parser = JsonOutputParser(pydantic_object=context_output_structure)
    prompt = """
    You're an expert code analysis assistant. You have received results from multiple search methods 
    analyzing a codebase based on the original query: "{original_query}"
    
    Here are the results from different subqueries:
    
    {aggregated_results}
    
    Synthesize these findings into a comprehensive, cohesive response that directly addresses 
    the original query. Organize your response according to the following structure:
    
    1. Knowledge: Provide a clear explanation and conclusion about the current issue based on the 
       aggregated results. Focus on what the code does, how it works, and any problems identified.
    
    2. Insights: Identify 2-3 key insights that would help solve or understand the issue better.
       These insights should point to specific functions, API endpoints, or files that are most 
       relevant to the query. Be precise and actionable.
    
    3. Code: If possible, provide any relevant code snippets or examples based on the context
       that would help address the original query. If no code can be provided based on the
       available information, you may leave this section empty.

       YOUR RESPONSE MUST BE VALID JSON with the following structure only:
    {
        "knowledge": "Clear explanation and conclusion about the current issue based on the aggregated results.",
        "insights": "2-3 key insights pointing to specific functions, API endpoints, or files that are most relevant.",
        "code": "Any relevant code snippets or examples that would help address the query. Leave empty if none available."
    }
    
    {format_instructions}
    """
    
    # Format the aggregated results for the prompt
    prompt_and_parser = PromptTemplate(
        template = prompt,
        input_variables = ['original_query', 'aggregated_results'],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
    )
    
    chain = prompt_and_parser | llm | parser
    result = chain.invoke({
        "original_query": state["original_query"],
        "aggregated_results": aggregated_data
        })
    print(f"THE DECOMPOSED QUERIES ARE ::: {result}")

    return {"final_result": result}

# Build the workflow with sequential flow
workflow.add_node("create_subqueries", create_subqueries)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("vectorCalls", vectorCalls)
workflow.add_node("graphCalls", graphCalls)
workflow.add_node("aggregateResults", aggregateResults)

# Set up sequential flow
workflow.add_edge(START, "create_subqueries")
workflow.add_edge("create_subqueries", "decompose_query")
workflow.add_edge("decompose_query", "vectorCalls")
workflow.add_edge("vectorCalls", "graphCalls")
workflow.add_edge("graphCalls", "aggregateResults")
workflow.add_edge("aggregateResults", END)

# Compile the workflow
app = workflow.compile()

# Example usage
def process_query(query: str):
    """Process a user query through the complete workflow."""
    print(f"Processing query: {query}")
    initial_state = {"original_query": query}
    result = app.invoke(initial_state)
    return result["final_result"]

# if __name__ == "__main__":
#     sample_query = "Find all API endpoints that don't have proper error handling"
#     result = process_query(sample_query)
#     print("\nFinal Result:")
#     print(result)
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator
from pinecone import Pinecone
import os

# === State Definition ===
class LangGraphState(TypedDict):
    original_query: str 
    subqueries: List[str]
    decomposed_queries: List[Dict[str, Any]]
    aggregated_data: List[Dict[str, Any]]
    final_result: Optional[str]

# === Database Connections === 
class PineconeOperation:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
    def retrieve_data(self, context: str) -> List[Dict]:
        # Implement actual Pinecone search
        return []

pc = PineconeOperation()

# === Pydantic Models ===
class SubqueriesModel(BaseModel):
    subqueries: List[str] = Field(description="Sub-queries derived from the main query")

class QueryDecomposition(BaseModel):
    subquery: str = Field(
        description="Specific aspect of the original query to investigate",
        example="Find all functions that call the payment gateway API"
    )
    intent_class: Literal['semantic', 'structural', 'hybrid'] = Field(
        description="Nature of the query: semantic (meaning) or structural (relationships)",
        example="structural"
    )
    scope: Dict[str, str] = Field(
        description="Target focus with type and specifics",
        example={
            "type": "class",
            "name": "PaymentService",
            "pattern": "api/gateways/*.py"
        }
    )
    operation: Literal['similarity_search', 'dependency_tracing', 'pattern_matching', 'path_finding'] = Field(
        description="Operation needed to resolve the subquery",
        example="dependency_tracing"
    )
    search_type: Literal['vector', 'graph', 'hybrid'] = Field(
        description="Search mechanism based on other properties",
        example="graph"
    )
    context_requirements: List[str] = Field(
        description="Additional context needed for execution",
        example=["API response formats", "Error handling patterns"]
    )

class DecomposerModel(BaseModel):
    queries: List[QueryDecomposition]

    @model_validator(mode="before")
    @classmethod
    def validate_queries(cls, values: Dict) -> Dict:
        queries = values.get("queries", [])
        if not queries:
            raise ValueError("At least one query decomposition is required")
        if len(queries) > 10:
            raise ValueError("Maximum 10 sub-queries allowed")
        return values

class ContextOutput(BaseModel):
    sub_query_output: str = Field(
        description="Concise output for the sub-query (1-3 sentences)",
        max_length=500
    )

    @model_validator(mode="before")
    @classmethod
    def validate_output(cls, values: Dict) -> Dict:
        output = values.get("sub_query_output", "")
        if len(output) > 500:
            raise ValueError("Output exceeds 500 character limit")
        if output.count(".") > 4:
            raise ValueError("Maximum 4 sentences allowed")
        return values

# === LangGraph Nodes ===
def create_subqueries(state: LangGraphState) -> LangGraphState:
    """Generate sub-queries from the original query"""
    parser = JsonOutputParser(pydantic_object=SubqueriesModel)
    
    prompt = PromptTemplate(
        template='''
        Analyze this query and break it into sub-queries:
        {query}
        
        {format_instructions}
        ''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"query": state["original_query"]})
    
    return {"subqueries": result.subqueries}

def decompose_query(state: LangGraphState) -> LangGraphState:
    """Break down sub-queries into structured operations"""
    parser = JsonOutputParser(pydantic_object=DecomposerModel)
    
    prompt = PromptTemplate(
        template='''
        Decompose this sub-query for code analysis:
        {sub_query}
        
        {format_instructions}
        ''',
        input_variables=["sub_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    decomposed = []
    for subq in state["subqueries"]:
        chain = prompt | llm | parser
        result = chain.invoke({"sub_query": subq})
        decomposed.extend([q.dict() for q in result.queries])
    
    return {"decomposed_queries": decomposed}

def vector_search(state: LangGraphState) -> LangGraphState:
    """Handle vector search operations"""
    results = []
    for query in state["decomposed_queries"]:
        if query["search_type"] in ("vector", "hybrid"):
            context = pc.retrieve_data(query["subquery"])
            results.append({
                "query": query["subquery"],
                "results": context
            })
    return {"aggregated_data": state["aggregated_data"] + results}

def graph_search(state: LangGraphState) -> LangGraphState:
    """Handle graph search operations"""
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username="neo4j",
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    results = []
    for query in state["decomposed_queries"]:
        if query["search_type"] in ("graph", "hybrid"):
            chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=True
            )
            result = chain.invoke(query["subquery"])
            results.append({
                "query": query["subquery"],
                "results": result["result"]
            })
    
    return {"aggregated_data": state["aggregated_data"] + results}

# === Workflow Setup ===
workflow = StateGraph(LangGraphState)

workflow.add_node("create_subqueries", create_subqueries)
workflow.add_node("decompose", decompose_query)
workflow.add_node("vector_search", vector_search)
workflow.add_node("graph_search", graph_search)

workflow.set_entry_point("create_subqueries")
workflow.add_edge("create_subqueries", "decompose")
workflow.add_edge("decompose", "vector_search")
workflow.add_edge("decompose", "graph_search")
workflow.add_edge("vector_search", END)
workflow.add_edge("graph_search", END)

app = workflow.compile()





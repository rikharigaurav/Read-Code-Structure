import os
import json
from typing import List, Dict, Any, Optional, Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, validator
from langchain_mistralai import ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from neo4j import GraphDatabase
import logging
import traceback
import time
from dotenv import load_dotenv

load_dotenv()

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubQuery(BaseModel):
    """A single subquery for issue investigation"""
    query_text: str = Field(..., description="The specific question or search intent")
    target_scope: str = Field(..., description="Code area to focus on (class, function, file, etc.)")
    operation_type: Literal["vector", "graph", "hybrid"] = Field(..., description="Type of search operation to perform")
    priority: int = Field(default=1, description="Priority order (1 being highest)")

class IssueMetadata(BaseModel):
    """Structured metadata about an issue"""
    title: str
    description: str
    steps_to_reproduce: Optional[List[str]] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    environment_details: Optional[Dict[str, str]] = None
    labels: Optional[List[str]] = None
    assignees: Optional[List[str]] = None
    milestone: Optional[str] = None
    
class CodeSnippet(BaseModel):
    """Relevant code snippet from the codebase"""
    file_path: str
    code: str
    relevance_score: float
    line_numbers: Optional[tuple] = None
    
class RelationshipData(BaseModel):
    """Data about code relationships from graph database"""
    relationship_type: str
    nodes: List[Dict[str, Any]]
    properties: Optional[Dict[str, Any]] = None
    cypher_query: Optional[str] = None
    
class SearchResult(BaseModel):
    """Result from either vector or graph search"""
    source_type: Literal["vector", "graph", "hybrid"]
    code_snippets: Optional[List[CodeSnippet]] = None
    relationships: Optional[List[RelationshipData]] = None
    insight: str = Field(..., description="LLM-generated insight from this search result")
    
class SolutionOutput(BaseModel):
    """Final output structure for issue solutions"""
    summary: str = Field(..., description="Brief English summary of the issue and solution")
    procedural_knowledge: Optional[str] = Field(None, description="Step-by-step instructions if applicable")
    code_solution: Optional[str] = Field(None, description="Generated code solution if applicable")
    visualization_query: Optional[str] = Field(None, description="Cypher query for visualization if applicable")

class IssueAnalyzerState(Dict[str, Any]):
    """State for the LangGraph workflow"""
    
    @classmethod
    def get_issue_metadata(cls, state: 'IssueAnalyzerState') -> Optional[IssueMetadata]:
        metadata = state.get("issue_metadata")
        logger.debug(f"Retrieved issue_metadata from state: {metadata is not None}")
        return metadata
    
    @classmethod
    def get_subqueries(cls, state: 'IssueAnalyzerState') -> List[SubQuery]:
        subqueries = state.get("subqueries", [])
        logger.debug(f"Retrieved {len(subqueries)} subqueries from state")
        return subqueries
    
    @classmethod
    def get_search_results(cls, state: 'IssueAnalyzerState') -> List[SearchResult]:
        results = state.get("search_results", [])
        logger.debug(f"Retrieved {len(results)} search results from state")
        return results
    
    @classmethod
    def get_solution(cls, state: 'IssueAnalyzerState') -> Optional[SolutionOutput]:
        solution = state.get("solution")
        logger.debug(f"Retrieved solution from state: {solution is not None}")
        return solution

# ============ Components ============

class IssueAnalyzer:
    """Components for analyzing and solving issues"""
    
    def __init__(self, 
                 llm_model: str = "codestral-latest",
                 neo4j_uri: str = None,
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 pinecone_api_key: str = None,
                 pinecone_index: str = None):
        
        logger.info(f"Initializing IssueAnalyzer with model: {llm_model}")
        
        # Initialize LLM
        try:
            self.llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'), model=llm_model)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize Neo4j
        self.neo4j_driver = None
        if neo4j_uri and neo4j_username and neo4j_password:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_username, neo4j_password)
                )
                logger.info("Neo4j driver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Initialize Pinecone
        self.vector_store = None
        if pinecone_api_key and pinecone_index:
            try:
                from pinecone import Pinecone
                Pinecone(api_key=pinecone_api_key)
                self.vector_store = PineconeVectorStore(index_name=pinecone_index, embedding=CohereEmbeddings(
                    cohere_api_key=os.getenv("COHERE_API_KEY_3"),
                    model="embed-english-v3.0"
                ))
                logger.info("Pinecone vector store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}")
                logger.error(traceback.format_exc())
            
    def run_vector_search(self, query: str, target_scope: str, limit: int = 5) -> List[CodeSnippet]:
        """Run vector search on the codebase"""
        logger.info(f"Running vector search for: {query} in scope: {target_scope}")
        
        if not self.vector_store:
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        # Add target scope to narrow down search
        enhanced_query = f"In {target_scope}: {query}"
        
        try:
            search_results = self.vector_store.similarity_search_with_score(
                enhanced_query, k=limit
            )
            
            code_snippets = []
            for doc, score in search_results:
                # Parse document metadata
                metadata = doc.metadata
                code_snippets.append(
                    CodeSnippet(
                        file_path=metadata.get("file_path", "unknown"),
                        code=doc.page_content,
                        relevance_score=float(score),
                        line_numbers=(metadata.get("start_line"), metadata.get("end_line"))
                    )
                )
            logger.info(f"Vector search returned {len(code_snippets)} snippets")
            return code_snippets
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def run_graph_query(self, query_intent: str, target_scope: str) -> List[RelationshipData]:
        """Run graph query on the codebase"""
        logger.info(f"Running graph query for: {query_intent} in scope: {target_scope}")
        
        if not self.neo4j_driver:
            logger.warning("Neo4j driver not initialized, returning empty results")
            return []
        
        # Generate appropriate Cypher query based on intent and scope
        cypher_query = self._generate_cypher_query(query_intent, target_scope)
        logger.info(f"Generated Cypher query: {cypher_query}")
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query)
                records = list(result)
                
                relationships = []
                for record in records:
                    # Process and structure the Neo4j results
                    rel_data = RelationshipData(
                        relationship_type=record.get("type", "unknown"),
                        nodes=[dict(node) for node in record.get("nodes", [])],
                        properties=record.get("properties"),
                        cypher_query=cypher_query
                    )
                    relationships.append(rel_data)
                logger.info(f"Graph query returned {len(relationships)} relationships")
                return relationships
        except Exception as e:
            logger.error(f"Graph query error: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _generate_cypher_query(self, query_intent: str, target_scope: str) -> str:
        """Generate appropriate Cypher query based on intent and scope"""
        # This is a simplified implementation
        # In practice, this would be more sophisticated, possibly using an LLM
        
        # Common query patterns based on intent
        if "dependency" in query_intent.lower():
            # Find dependencies for a specific component
            return f"""
            MATCH (n:CodeEntity)-[r:DEPENDS_ON]->(m:CodeEntity)
            WHERE n.name CONTAINS '{target_scope}' OR m.name CONTAINS '{target_scope}'
            RETURN r.type as type, [n, m] as nodes, r as properties
            LIMIT 10
            """
        elif "call" in query_intent.lower() or "invoke" in query_intent.lower():
            # Find function calls
            return f"""
            MATCH (caller:Function)-[r:CALLS]->(callee:Function)
            WHERE caller.name CONTAINS '{target_scope}' OR callee.name CONTAINS '{target_scope}'
            RETURN r.type as type, [caller, callee] as nodes, r as properties
            LIMIT 10
            """
        elif "import" in query_intent.lower():
            # Find import relationships
            return f"""
            MATCH (f:File)-[r:IMPORTS]->(m:Module)
            WHERE f.path CONTAINS '{target_scope}' OR m.name CONTAINS '{target_scope}'
            RETURN r.type as type, [f, m] as nodes, r as properties
            LIMIT 10
            """
        elif "error" in query_intent.lower() or "exception" in query_intent.lower():
            # Find error handling
            return f"""
            MATCH (f:Function)-[r:HANDLES_EXCEPTION]->(e:ExceptionType)
            WHERE f.name CONTAINS '{target_scope}'
            RETURN r.type as type, [f, e] as nodes, r as properties
            LIMIT 10
            """
        else:
            # Generic code relationship query
            return f"""
            MATCH (n)-[r]->(m)
            WHERE n.name CONTAINS '{target_scope}' OR m.name CONTAINS '{target_scope}'
            RETURN type(r) as type, [n, m] as nodes, properties(r) as properties
            LIMIT 10
            """
            
    def analyze_issue(self, issue_text: str) -> IssueMetadata:
        """Analyze issue text to extract structured metadata"""
        logger.info("Analyzing issue metadata")
        logger.debug(f"Issue text length: {len(issue_text)}")
        
        prompt = f"""
        Analyze the following issue and extract structured metadata.
        
        Issue:
        {issue_text}
        
        Extract and organize the following:
        1. Issue title (make it descriptive and action-oriented)
        2. Clear problem description
        3. Steps to reproduce (if provided)
        4. Expected behavior
        5. Actual behavior
        6. Environment details
        7. Labels (e.g., bug, enhancement, etc.)
        8. Assigned contributors (if mentioned)
        9. Milestone or timeline information (if mentioned)
        
        Format your response as a JSON object with the following structure:
        {{
            "title": "...",
            "description": "...",
            "steps_to_reproduce": ["step 1", "step 2", ...],
            "expected_behavior": "...",
            "actual_behavior": "...",
            "environment_details": {{...}},
            "labels": ["label1", "label2", ...],
            "assignees": ["person1", "person2", ...],
            "milestone": "..."
        }}
        
        Ensure that your response is ONLY the JSON object, nothing else.
        """
        
        try:
            logger.debug("Sending issue analysis prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"Received response from LLM: {response.content[:100]}...")
            
            metadata_json = self._extract_json_from_response(response.content)
            logger.debug(f"Extracted JSON: {json.dumps(metadata_json)[:100]}...")
            
            issue_metadata = IssueMetadata(**metadata_json)
            logger.info(f"Successfully created IssueMetadata object with title: {issue_metadata.title}")
            return issue_metadata
        except Exception as e:
            logger.error(f"Issue analysis error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return basic metadata if analysis fails
            logger.warning("Falling back to basic metadata")
            return IssueMetadata(
                title="Issue Analysis Failed",
                description=issue_text[:200] + "..." if len(issue_text) > 200 else issue_text
            )
    
    def create_subqueries(self, issue_metadata: IssueMetadata) -> List[SubQuery]:
        """Break down the issue into specific subqueries"""
        logger.info("Creating subqueries for issue investigation")
        
        prompt = f"""
        Based on the following issue metadata, create a list of specific subqueries for investigation.
        
        Issue Metadata:
        Title: {issue_metadata.title}
        Description: {issue_metadata.description}
        Steps to Reproduce: {issue_metadata.steps_to_reproduce if issue_metadata.steps_to_reproduce else 'N/A'}
        Expected Behavior: {issue_metadata.expected_behavior if issue_metadata.expected_behavior else 'N/A'}
        Actual Behavior: {issue_metadata.actual_behavior if issue_metadata.actual_behavior else 'N/A'}
        Environment: {issue_metadata.environment_details if issue_metadata.environment_details else 'N/A'}
        Labels: {', '.join(issue_metadata.labels) if issue_metadata.labels else 'N/A'}
        
        For each subquery, provide:
        1. query_text: A specific question or search intent
        2. target_scope: Code area to focus on (class, function, file, etc.)
        3. operation_type: Either "vector" (for semantic search), "graph" (for relationship analysis), or "hybrid" (for both)
        4. priority: Priority order (1 being highest)
        
        Create 3-5 subqueries that will help understand and solve this issue.
        Format your response as a JSON list of objects with the following structure:
        [
            {{
                "query_text": "...",
                "target_scope": "...",
                "operation_type": "...",
                "priority": ...
            }},
            ...
        ]
        
        Ensure that your response is ONLY the JSON list, nothing else.
        """
        
        try:
            logger.debug("Sending subquery creation prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"Received response from LLM: {response.content[:100]}...")
            
            subqueries_json = self._extract_json_from_response(response.content)
            logger.debug(f"Extracted JSON: {json.dumps(subqueries_json)[:100]}...")
            
            subqueries = [SubQuery(**sq) for sq in subqueries_json]
            logger.info(f"Successfully created {len(subqueries)} SubQuery objects")
            return subqueries
        except Exception as e:
            logger.error(f"Subquery creation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return basic subquery if creation fails
            logger.warning("Falling back to basic subquery")
            return [
                SubQuery(
                    query_text=f"Find code related to {issue_metadata.title}",
                    target_scope="main functionality",
                    operation_type="vector",
                    priority=1
                )
            ]
    
    def process_subquery(self, subquery: SubQuery) -> SearchResult:
        """Process a single subquery using appropriate search method"""
        logger.info(f"Processing subquery: {subquery.query_text} using {subquery.operation_type}")
        
        result = SearchResult(source_type=subquery.operation_type, insight="")
        
        try:
            if subquery.operation_type == "vector":
                # Perform vector search
                code_snippets = self.run_vector_search(
                    subquery.query_text, 
                    subquery.target_scope
                )
                result.code_snippets = code_snippets
                
            elif subquery.operation_type == "graph":
                # Perform graph search
                relationships = self.run_graph_query(
                    subquery.query_text,
                    subquery.target_scope
                )
                result.relationships = relationships
                
            elif subquery.operation_type == "hybrid":
                # Perform both vector and graph search
                code_snippets = self.run_vector_search(
                    subquery.query_text, 
                    subquery.target_scope
                )
                relationships = self.run_graph_query(
                    subquery.query_text,
                    subquery.target_scope
                )
                result.code_snippets = code_snippets
                result.relationships = relationships
            
            # Generate insight from search results
            result.insight = self._generate_insight_from_results(subquery, result)
            logger.info(f"Completed processing subquery with insight length: {len(result.insight)}")
            return result
            
        except Exception as e:
            logger.error(f"Subquery processing error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty result with error message
            return SearchResult(
                source_type=subquery.operation_type,
                insight=f"Error processing subquery: {str(e)}"
            )
    
    def _generate_insight_from_results(self, subquery: SubQuery, result: SearchResult) -> str:
        """Generate insights from search results"""
        
        # Prepare context from code snippets
        code_context = ""
        if result.code_snippets:
            code_context = "\n\n".join([
                f"File: {snippet.file_path}\n```\n{snippet.code}\n```" 
                for snippet in result.code_snippets[:3]  # Limit to top 3 for context
            ])
        
        # Prepare context from relationships
        relationship_context = ""
        if result.relationships:
            relationship_context = "\n\n".join([
                f"Relationship: {rel.relationship_type}\nNodes: {', '.join([str(n) for n in rel.nodes[:2]])}"
                for rel in result.relationships[:3]  # Limit to top 3 for context
            ])
        
        prompt = f"""
        Based on the following code search results, generate an insight that answers the query:
        
        Query: {subquery.query_text}
        Target Scope: {subquery.target_scope}
        
        Code Snippets:
        {code_context if code_context else "No code snippets found."}
        
        Code Relationships:
        {relationship_context if relationship_context else "No relationships found."}
        
        Provide a concise insight (2-4 sentences) that explains what you found and how it relates to the query.
        Focus on technical details and implications for solving the issue.
        """
        
        try:
            logger.debug("Sending insight generation prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"Received insight from LLM: {response.content[:100]}...")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Insight generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return "Could not generate insight due to an error."
    
    def generate_solution(self, issue_metadata: IssueMetadata, search_results: List[SearchResult]) -> SolutionOutput:
        """Generate a comprehensive solution based on all search results"""
        logger.info("Generating solution from search results")
        logger.debug(f"Working with {len(search_results)} search results")
        
        # Prepare context from all search results
        results_context = "\n\n".join([
            f"Subquery Result {i+1}:\n{result.insight}" 
            for i, result in enumerate(search_results)
        ])
        
        # Prepare code snippets context
        code_snippets_context = ""
        for result in search_results:
            if result.code_snippets:
                for snippet in result.code_snippets:
                    code_snippets_context += f"\nFile: {snippet.file_path}\n```\n{snippet.code}\n```\n"
        
        # Prepare relationships context
        relationships_context = ""
        visualization_query = None
        for result in search_results:
            if result.relationships and len(result.relationships) > 0:
                # Use the first relationship's cypher query for visualization if available
                if not visualization_query and result.relationships[0].cypher_query:
                    visualization_query = result.relationships[0].cypher_query
                
                for rel in result.relationships:
                    relationships_context += f"\nRelationship: {rel.relationship_type}\n"
                    relationships_context += f"Nodes: {', '.join([str(n) for n in rel.nodes[:2]])}\n"
        
        prompt = f"""
        Based on the issue and search results, generate a comprehensive solution.
        
        Issue:
        Title: {issue_metadata.title}
        Description: {issue_metadata.description}
        Expected Behavior: {issue_metadata.expected_behavior if issue_metadata.expected_behavior else 'N/A'}
        Actual Behavior: {issue_metadata.actual_behavior if issue_metadata.actual_behavior else 'N/A'}
        
        Search Results Summary:
        {results_context}
        
        Relevant Code:
        {code_snippets_context[:1000] if code_snippets_context else "No code snippets available."}
        
        Code Relationships:
        {relationships_context[:500] if relationships_context else "No relationships available."}
        
        Generate a solution with the following components:
        1. A brief English summary of the issue and solution (2-3 paragraphs)
        2. Procedural knowledge for solving the issue (step-by-step instructions)
        3. Code solution (if applicable)
        
        Format your response as a JSON object with the following structure:
        {{
            "summary": "...",
            "procedural_knowledge": "...",
            "code_solution": "...",
            "visualization_query": "..."
        }}
        
        For the code solution, focus on being precise, correct, and addressing the root cause.
        If no code solution is needed, set that field to null.
        For the visualization query, use Cypher syntax compatible with Neo4j.
        
        Ensure that your response is ONLY the JSON object, nothing else.
        """
        
        try:
            logger.debug("Sending solution generation prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"Received solution from LLM: {response.content[:100]}...")
            
            solution_json = self._extract_json_from_response(response.content)
            logger.debug(f"Extracted solution JSON: {json.dumps(solution_json)[:100]}...")
            
            # Add visualization query if available
            if visualization_query and not solution_json.get("visualization_query"):
                solution_json["visualization_query"] = visualization_query
                
            solution = SolutionOutput(**solution_json)
            logger.info(f"Successfully created SolutionOutput with summary length: {len(solution.summary)}")
            return solution
        except Exception as e:
            logger.error(f"Solution generation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return basic solution if generation fails
            return SolutionOutput(
                summary=f"Failed to generate complete solution due to error: {str(e)}",
                procedural_knowledge="Not available due to error.",
                code_solution=None,
                visualization_query=None
            )
    
    def _extract_json_from_response(self, text: str) -> Any:
        """Extract JSON from LLM response"""
        try:
            logger.debug(f"Attempting to extract JSON from text of length {len(text)}")
            # Find JSON in the text (handling cases where LLM adds explanations)
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx == -1 and text.find('[') != -1:
                # Try finding array instead
                start_idx = text.find('[')
                end_idx = text.rfind(']')
            
            logger.debug(f"JSON markers: start_idx={start_idx}, end_idx={end_idx}")
                
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                logger.debug(f"Extracted JSON string of length {len(json_str)}")
                parsed_json = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON object")
                return parsed_json
            else:
                # If no JSON structure found, attempt to parse the whole text
                logger.debug("No clear JSON markers found, attempting to parse entire text")
                parsed_json = json.loads(text)
                logger.debug(f"Successfully parsed entire text as JSON")
                return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Problematic text: {text[:200]}...")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to parse JSON from response: {str(e)}")

# ============ LangGraph Flow ============

def build_issue_solver_graph(analyzer: IssueAnalyzer) -> StateGraph:
    """Build the workflow graph for issue solving"""
    logger.info("Building issue solver workflow graph")
    print("Entering build_issue_solver_graph")
    
    class IssueWorkflowState(IssueAnalyzerState):   
        issue_text: Optional[str] = None
        issue_metadata: Optional[IssueMetadata] = None
        subqueries: List[SubQuery] = []
        search_results: List[SearchResult] = []
        solution: Optional[SolutionOutput] = None
        final_output: Optional[SolutionOutput] = None
        error: Optional[str] = None
    
    # Define the state graph with the schema
    workflow = StateGraph(IssueWorkflowState)
    
    # Define the nodes as functions
    def analyze_issue(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for analyzing the issue text"""
        logger.info("CHECKPOINT: Entering analyze_issue node")
        
        issue_text = state.get("issue_text", "")
        logger.debug(f"Retrieved issue_text from state, length: {len(issue_text)}")
        print(f"Issue text exists: {bool(issue_text)}, length: {len(issue_text)}")
        
        if not issue_text:
            logger.error("No issue text provided")
            return {"error": "No issue text provided"}
        
        issue_metadata = analyzer.analyze_issue(issue_text)
        logger.debug(f"Created issue_metadata with title: {issue_metadata.title}")
        
        # Debug the state we're returning
        result = {"issue_metadata": issue_metadata}
        logger.debug(f"Returning from analyze_issue with keys: {list(result.keys())}")
        print(f"analyze_issue returns: {list(result.keys())}")
        
        return result
    
    def create_subqueries(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for creating subqueries based on issue metadata"""
        logger.info("CHECKPOINT: Entering create_subqueries node")
        print("Entering create_subqueries")
        
        issue_metadata = IssueAnalyzerState.get_issue_metadata(state)
        if not issue_metadata:
            error_msg = "No issue metadata available"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        logger.debug(f"Retrieved issue_metadata with title: {issue_metadata.title}")
        print(f"Issue metadata title: {issue_metadata.title}")
            
        subqueries = analyzer.create_subqueries(issue_metadata)
        logger.debug(f"Created {len(subqueries)} subqueries")
        
        # Debug the state we're returning
        result = {"subqueries": subqueries}
        logger.debug(f"Returning from create_subqueries with keys: {list(result.keys())}")
        print(f"create_subqueries returns: {list(result.keys())}")
        
        return result
    
    def process_subqueries(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for processing all subqueries"""
        logger.info("CHECKPOINT: Entering process_subqueries node")
        print("Entering process_subqueries")
        
        subqueries = IssueAnalyzerState.get_subqueries(state)
        if not subqueries:
            error_msg = "No subqueries available"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        logger.debug(f"Retrieved {len(subqueries)} subqueries")
        print(f"Processing {len(subqueries)} subqueries")
            
        # Sort subqueries by priority
        sorted_subqueries = sorted(subqueries, key=lambda sq: sq.priority)
        
        # Process each subquery
        search_results = []
        for i, subquery in enumerate(sorted_subqueries):
            logger.debug(f"Processing subquery {i+1}/{len(sorted_subqueries)}: {subquery.query_text}")
            print(f"Processing subquery {i+1}/{len(sorted_subqueries)}")
            result = analyzer.process_subquery(subquery)
            search_results.append(result)
        
        # Debug the state we're returning
        result = {"search_results": search_results}
        logger.debug(f"Returning from process_subqueries with keys: {list(result.keys())}")
        print(f"process_subqueries returns: {list(result.keys())} with {len(search_results)} results")
        return result
    
    def generate_solution(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for generating the final solution"""
        logger.info("Entering generate_solution node")
        
        issue_metadata = IssueAnalyzerState.get_issue_metadata(state)
        search_results = IssueAnalyzerState.get_search_results(state)
        
        if not issue_metadata:
            logger.error("Missing issue metadata for solution generation")
            return {"error": "Missing required data for solution generation"}
            
        if not search_results:
            logger.error("Missing search results for solution generation")
            return {"error": "Missing required data for solution generation"}
            
        logger.debug(f"Issue metadata: {issue_metadata}")
        logger.debug(f"Search results count: {len(search_results)}")
        
        try:
            logger.info("Generating solution from analyzer")
            solution = analyzer.generate_solution(issue_metadata, search_results)
            logger.info("Solution generated successfully")
            return {"solution": solution}
        except Exception as e:
            logger.exception(f"Error generating solution: {str(e)}")
            return {"error": f"Exception in solution generation: {str(e)}"}

    def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for formatting the final output"""
        logger.info("Entering format_output node")
        
        solution = IssueAnalyzerState.get_solution(state)
        if not solution:
            logger.error("No solution available for formatting")
            return {"error": "No solution available"}
            
        logger.debug(f"Solution to format: {solution}")
        
        # Return the state update
        return {"final_output": solution}
    
    # Add nodes to the graph
    logger.debug("Adding nodes to workflow")
    workflow.add_node("analyze_issue", analyze_issue)
    workflow.add_node("create_subqueries", create_subqueries)
    workflow.add_node("process_subqueries", process_subqueries)
    workflow.add_node("generate_solution", generate_solution)
    workflow.add_node("format_output", format_output)
    
    # Build the graph edges
    workflow.add_edge("analyze_issue", "create_subqueries")
    workflow.add_edge("create_subqueries", "process_subqueries")
    workflow.add_edge("process_subqueries", "generate_solution")
    workflow.add_edge("generate_solution", "format_output")
    workflow.add_edge("format_output", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze_issue")
    
    return workflow

# ============ Application Setup ============

def create_issue_solver(
    llm_model: str = "codestral-latest",
    neo4j_uri: str = None,
    neo4j_username: str = None,
    neo4j_password: str = None,
    pinecone_api_key: str = None,
    pinecone_index: str = None,
    log_level: int = logging.INFO
) -> callable:
    """Create and return the issue solver function with configurable logging"""
    # Setup logging
    logger.setLevel(log_level)
    logger.info(f"Creating issue solver with model: {llm_model}")
    
    # Log config info without sensitive data
    logger.info(f"Neo4j configured: {bool(neo4j_uri and neo4j_username)}")
    logger.info(f"Pinecone configured: {bool(pinecone_api_key and pinecone_index)}")
    
    try:
        # Initialize the analyzer
        logger.info("Initializing issue analyzer")
        analyzer = IssueAnalyzer(
            llm_model=llm_model,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index
        )
        
        # Build the workflow graph
        logger.info("Building issue solver workflow")
        workflow = build_issue_solver_graph(analyzer)
        
        # Compile the graph
        logger.info("Compiling workflow")
        app = workflow.compile()
        logger.info("Workflow compiled successfully")
        
    except Exception as e:
        logger.exception(f"Error during issue solver initialization: {str(e)}")
        raise
    
    def solve_issue(issue_text: str) -> SolutionOutput:
        """Solve an issue based on the provided description"""
        issue_summary = issue_text[:100] + "..." if len(issue_text) > 100 else issue_text
        logger.info(f"Processing new issue: {issue_summary}")
        
        try:
            # Run the workflow with the issue text in the initial state
            logger.info("Invoking workflow")
            start_time = time.time()
            
            initial_state = {"issue_text": issue_text}
            logger.debug(f"Initial state keys: {list(initial_state.keys())}")
            
            result = app.invoke(initial_state)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")
            logger.debug(f"Result keys: {list(result.keys())}")
            
            # Extract the solution from the final state
            if "final_output" in result:
                logger.info("Solution generated successfully")
                return result["final_output"]
            elif "solution" in result:
                logger.info("Solution found in result")
                return result["solution"]
            elif "error" in result:
                error_msg = f"Error in processing: {result['error']}"
                logger.error(error_msg)
                return SolutionOutput(
                    summary=error_msg,
                    procedural_knowledge=None,
                    code_solution=None,
                    visualization_query=None
                )
            else:
                error_msg = "Unknown error occurred during processing"
                logger.error(error_msg)
                return SolutionOutput(
                    summary=error_msg,
                    procedural_knowledge=None,
                    code_solution=None,
                    visualization_query=None
                )
                
        except Exception as e:
            logger.exception(f"Issue solving error: {str(e)}")
            return SolutionOutput(
                summary=f"Exception occurred: {str(e)}",
                procedural_knowledge=None,
                code_solution=None,
                visualization_query=None
            )
    
    return solve_issue

# Example usage
if __name__ == "__main__":
    # Set up environment variables
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_username = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index = os.environ.get("PINECONE_INDEX")
    
    # Get log level from environment or default to INFO
    log_level_name = os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Create the issue solver with logging
    issue_solver = create_issue_solver(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        pinecone_api_key=pinecone_api_key,
        pinecone_index=pinecone_index,
        log_level=log_level
    )
    
    # Example issue text
    example_issue = """
    Title: API returning 500 error on pagination request
    
    Description:
    When trying to paginate through search results using the search API, it works for the first 3 pages but then returns a 500 error on page 4 and beyond.
    
    Steps to reproduce:
    1. Call GET /api/search?q=test
    2. Get the first page of results
    3. Follow the "next" link 3 times
    4. Try to access the 4th page
    
    Expected behavior:
    All pages should return results with 200 OK
    
    Actual behavior:
    Pages 1-3 work fine, but page 4 returns a 500 Internal Server Error
    
    Error message from logs:
    "RangeError: Maximum call stack size exceeded at Object.paginate (/src/utils/pagination.js:42:19)"
    
    Environment:
    Production server
    API version: 2.3.1
    """
    
    # # Solve the issue with comprehensive logging
    # logger.info("Starting issue solving process")
    # solution = issue_solver(example_issue)
    # logger.info("Issue solving completed")
    
    # # Print the solution
    # print(json.dumps(solution.dict(), indent=2))
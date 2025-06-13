import os
import json
from typing import List, Dict, Any, Optional, Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, validator
from langchain_mistralai import ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from neo4j import GraphDatabase
from utils.pinecone_db import pinecone
import logging
import traceback
import time
from dotenv import load_dotenv

load_dotenv()

# logging.basicConfig(
#     level=logging.DEBUG,  
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

class CypherQueryOutput(BaseModel):
    cypher_query: str

class GraphRetrieval(BaseModel):
    insight: str = Field(..., description="LLM-generated insight from this search result"),
    relationships: Optional[List[str]] = Field(..., description="List of relationships found in the graph database"),

class VectorRetrieval(BaseModel):
    insights: str = Field(..., description="How the file content and metadata help solve the query and achieve the stated purpose")
    relevant_code_sections: List[str] = Field(..., description="Specific code elements most relevant to resolving the query")

class SubQuery(BaseModel):
    """A single subquery for issue investigation"""
    query_text: str = Field(..., description="Detailed context and keywords for retrieval or graph traversal")
    namespace: Optional[str] = Field(None, description="Specify the namespace from the Pinecone index to search in vector search")
    operation_type: Literal["vector", "graph"] = Field(..., description="Type of search operation to perform")
    purpose: str = Field(..., description="What this query targets - the specific aspect of the issue being investigated")



class IssueMetadata(BaseModel):
    """Structured metadata about an issue"""
    title: str = Field(..., description="Descriptive and action-oriented issue title")
    description: str = Field(..., description="Comprehensive description including all technical details")
    steps_to_reproduce: Optional[List[str]] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    environment_details: Optional[Dict[str, str]] = None
    labels: Optional[List[str]] = None
    
# class CodeSnippet(BaseModel):
#     """Relevant code snippet from the codebase"""
#     file_path: str
#     code: str
#     relevance_score: float
#     line_numbers: Optional[tuple] = None

class RelationshipData(BaseModel):
    """Data about code relationships from graph database"""
    relationship_type: str
    nodes: List[Dict[str, Any]]
    properties: Optional[Dict[str, Any]] = None
    cypher_query: Optional[str] = None
    
class SearchResult(BaseModel):
    """Result from either vector or graph search"""
    source_type: Literal["vector", "graph"]
    code_snippets: Optional[List[str]] = None
    relationships: Optional[List[str]] = None
    insight: str = Field(..., description="LLM-generated insight from this search result")
    
class SolutionOutput(BaseModel):
    """Final output structure for issue solutions"""
    summary: str = Field(..., description="Brief English summary of the issue and solution")
    procedural_knowledge: Optional[list[str]] = Field(None, description="Step-by-step instructions if applicable")
    code_solution: Optional[str] = Field(None, description="Generated code solution if applicable")
    visualization_query: Optional[str] = Field(None, description="Cypher query for visualization if applicable")

class IssueAnalyzerState(Dict[str, Any]):
    """State for the LangGraph workflow"""
    
    @classmethod
    def get_issue_metadata(cls, state: 'IssueAnalyzerState') -> Optional[IssueMetadata]:
        metadata = state.get("issue_metadata")
        #logger.debug(f"Retrieved issue_metadata from state: {metadata is not None}")
        return metadata
    
    @classmethod
    def get_subqueries(cls, state: 'IssueAnalyzerState') -> List[SubQuery]:
        subqueries = state.get("subqueries", [])
        #logger.debug(f"Retrieved {len(subqueries)} subqueries from state")
        return subqueries
    
    @classmethod
    def get_search_results(cls, state: 'IssueAnalyzerState') -> List[SearchResult]:
        results = state.get("search_results", [])
        #logger.debug(f"Retrieved {len(results)} search results from state")
        return results
    
    @classmethod
    def get_solution(cls, state: 'IssueAnalyzerState') -> Optional[SolutionOutput]:
        solution = state.get("solution")
        #logger.debug(f"Retrieved solution from state: {solution is not None}")
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
        
        #logger.info(f"Initializing IssueAnalyzer with model: {llm_model}")
        
        # Initialize LLM
        try:
            self.llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY_1'), model=llm_model)
            #logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            #logger.error(traceback.format_exc())
            raise
        
        # Initialize Neo4j
        self.neo4j_driver = None
        if neo4j_uri and neo4j_username and neo4j_password:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_username, neo4j_password)
                )
                #logger.info("Neo4j driver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
    
    def run_graph_query(self, query_intent: str ,purpose : str) -> List[RelationshipData]:
        """Run graph query on the codebase"""
        if not self.neo4j_driver:
            logger.warning("Neo4j driver not initialized, returning empty results")
            return []
    
        # Generate appropriate Cypher query based on intent and scope
        cypher_query = self._generate_cypher_query(query_intent)
        #logger.info(f"Generated Cypher query: {cypher_query}")
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query)
                records = list(result)
                print(f"the records found {records}")   
                
                # relationships = []
                # for record in records:
                #     # Process and structure the Neo4j results
                #     rel_data = RelationshipData(
                #         relationship_type=record.get("type", "unknown"),
                #         nodes=[dict(node) for node in record.get("nodes", [])],
                #         properties=record.get("properties"),
                #         cypher_query=cypher_query
                #     )
                #     relationships.append(rel_data)
                #logger.info(f"Graph query returned {len(relationships)} relationships")
                return {
                    'cypher_query': cypher_query,
                    'records': records
                }
        except Exception as e:
            logger.error(f"Graph query error: {str(e)}")
            #logger.error(traceback.format_exc())
            return []
    
    def _generate_cypher_query(self, query_intent: str) -> str:

        #logger.info(f"Generating Cypher query for intent: {query_intent} in scope: {target_scope}")
        
        parser = JsonOutputParser(pydantic_object=CypherQueryOutput)
        prompt = f"""
        Generate a Neo4j Cypher query that addresses the following query intent within the specified code scope:
        
        Query Intent: {query_intent}
        
        Knowledge Base Structure:
        
        NODES:
        - DataFile: id, file_name, file_path, file_ext, summary
        - TemplateMarkupFile: id, file_name, file_path, file_ext, summary
        - TestingFile: id, file_name, file_path, file_ext, test_framework, test_reference_dict, summary
        - DocumentationFile: id, file_name, file_path, file_ext, summary
        - APIEndpoint: id, endpoint (URL path), http_method (GET, POST, etc.), summary
        - Function: id, function_name, file_path, return_type, summary
        - Folder: id, folder_name, directory_path
        
        RELATIONSHIPS:
        - BELONGS_TO: Connects child nodes to parent nodes (e.g., file->folder, function->file)
        - CALLS: Connects functions that call other functions (e.g., file->functionB)
        - TEST: Connects test files to the functions or API endpoints they test
        - All HTTP methods (GET, POST, PUT, DELETE): Connect API routes to API endpoints
        
        Instructions:
        0. File CALLS function
        1. Analyze the query intent to understand what the user is looking for
        2. Generate a SINGLE efficient Cypher query that addresses the query intent within the specified code scope
        3. Use the appropriate node labels and properties from the knowledge base structure
        4. Use the appropriate relationship types as defined in the knowledge base structure
        5. Limit results to a reasonable amount (max 10 nodes or relationships unless otherwise specified)
        6. For complex queries, consider using pattern recognition or path finding 
        7. Always return the relationships between nodes (MOST IMPORTANT)
        
        Common query patterns:
        - Finding dependencies between components
        - Tracing function call hierarchies
        - Identifying which test files cover specific functions
        - Exploring file organization within folders
        - don't make the cypher query too complex
        - just simple relationships and there properties
        - Analyzing API endpoint usage and connections
        - Always ask for relationships between nodes (MOST IMPORTANT)
        
        Return ONLY a JSON object with a single key 'cypher_query' containing the Cypher query as a string.
        
        Example output format:
        {{
        "cypher_query": "MATCH p = (f:DataFile)-[]->(g:Function) WHERE g.function_name = 'get' RETURN p, r LIMIT 10;"
        }}
        """
        
        try:
            #logger.debug("Sending Cypher query generation prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            #logger.debug(f"Received response from LLM: {response.content[:100]}...")
            print("the graph CQL")
            print(response)
            
            # Parse the response to extract the Cypher query
            parsed_output = parser.parse(response.content)
            cypher_query = parsed_output['cypher_query']
            
            #logger.info(f"Successfully generated Cypher query: {cypher_query[:50]}...")
            return cypher_query
            
        except Exception as e:
            logger.error(f"Error generating Cypher query: {str(e)}")
            #logger.error(traceback.format_exc())
            
            # fallback_query = f"""
            # MATCH (n) 
            # WHERE (n:DataFile OR n:TemplateMarkupFile OR n:TestingFile OR n:DocumentationFile OR n:Function OR n:APIEndpoint) 
            # AND (n.file_name CONTAINS '{target_scope}' OR n.function_name CONTAINS '{target_scope}' OR n.endpoint CONTAINS '{target_scope}')
            # RETURN n LIMIT 5
            # """
            # #logger.warning(f"Using fallback Cypher query: {fallback_query}")
            # return fallback_query
    
    def analyze_issue(self, issue_text: str) -> IssueMetadata:
        #logger.info("Analyzing issue metadata")
        #logger.debug(f"Issue text length: {len(issue_text)}")
        
        prompt = f"""
        You are an expert software engineer analyzing a GitHub issue. Your task is to extract and structure comprehensive metadata that will be used for intelligent code investigation and problem-solving.

        Issue Content:
        {issue_text}

        ANALYSIS REQUIREMENTS:

        1. TITLE CREATION:
        - Create a clear, specific, and action-oriented title
        - Include the main component/feature affected
        - Use technical terminology when appropriate
        - Format: "[Component/Feature]: Brief description of the problem/request"
        - Examples: "Authentication API: Token validation fails with expired JWT", "Database Migration: Foreign key constraint error in user_profiles table"

        2. COMPREHENSIVE DESCRIPTION:
        - Provide an exhaustive, detailed description that captures EVERY aspect of the issue
        - Include ALL code snippets, error messages, stack traces, and technical details EXACTLY as provided
        - Preserve code formatting and structure
        - Add technical context and implications
        - Explain the business impact and user experience effects
        - Include relevant technical background and system interactions
        - Mention affected components, modules, or services
        - NO length restrictions - be as detailed as necessary for complete understanding
        - Structure the description with clear sections if the issue is complex

        3. STEPS TO REPRODUCE:
        - Extract clear, sequential steps that led to the issue
        - Include specific inputs, configurations, or conditions
        - Add technical details like API endpoints, parameters, or environment setup
        - Make steps actionable and precise

        4. EXPECTED vs ACTUAL BEHAVIOR:
        - Expected: What should happen according to specifications, documentation, or user expectations
        - Actual: What actually occurs, including all symptoms, error messages, and unexpected outcomes
        - Be specific about differences and their technical implications

        5. ENVIRONMENT DETAILS:
        - Extract all technical environment information mentioned
        - Include versions, operating systems, browsers, databases, frameworks
        - Add configuration details, deployment environment, or infrastructure specifics
        - Structure as key-value pairs for easy reference

        6. INTELLIGENT LABELING:
        - Assign appropriate labels based on issue analysis:
        * Problem type: bug, enhancement, feature-request, performance, security, documentation
        * Severity: critical, high, medium, low
        * Components: api, frontend, backend, database, authentication, ui-ux, testing
        * Technical areas: performance, security, accessibility, compatibility
        * Workflow: needs-investigation, ready-for-development, blocked
        - Use 3-6 most relevant labels

        CRITICAL INSTRUCTIONS:
        - Preserve ALL code snippets, error messages, and technical details in the description
        - Make the description comprehensive enough that someone can understand the full context without reading the original issue
        - Use technical language appropriately and explain complex concepts
        - Structure information logically with clear sections if needed
        - Ensure accuracy - don't add information not present in the original issue

        Format your response as a JSON object with this exact structure:
        {{
            "title": "Specific, action-oriented title with component context",
            "description": "Comprehensive, detailed description including all code, errors, and technical context. Structure with sections if complex. Include every relevant detail for complete understanding.",
            "steps_to_reproduce": ["Detailed step 1 with technical specifics", "Detailed step 2 with parameters/configs", "..."],
            "expected_behavior": "Detailed explanation of what should happen, including technical specifications",
            "actual_behavior": "Detailed explanation of what actually happens, including all symptoms and error details", 
            "environment_details": {{
                "framework_version": "...",
                "operating_system": "...",
                "browser": "...",
                "database": "...",
                "deployment": "...",
                "other_relevant_details": "..."
            }},
            "labels": ["intelligent", "contextual", "labels", "based", "on", "analysis"]
        }}

        IMPORTANT: Your response must be ONLY the JSON object, nothing else. Ensure all JSON strings are properly escaped.
        """
        
        try:
            parser = JsonOutputParser(pydantic_object=IssueMetadata)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            issue_metadata = parser.parse(response.content)
            print(f"Issue metadata: {issue_metadata}")
            return issue_metadata
            
        except Exception as e:
            logger.error(f"Issue analysis error: {str(e)}")
            #logger.error(traceback.format_exc())
            # Return more informative fallback metadata
            #logger.warning("Falling back to enhanced basic metadata")
            return IssueMetadata(
                title="Issue Analysis Failed - Manual Review Required",
                description=f"AUTOMATED ANALYSIS FAILED - Original Issue Content:\n\n{issue_text}\n\nThis issue requires manual analysis to extract proper metadata. The original content has been preserved above for reference.",
                labels=["analysis-failed", "manual-review-needed"]
            )
    
    def create_subqueries(self, issue_metadata: IssueMetadata) -> List[SubQuery]:
        namespace_list = pinecone.get_namespace_names()
        prompt = f"""
        You are helping to investigate a GitHub issue by breaking it down into focused subqueries. 
        Each subquery targets a specific aspect of understanding and solving the issue.

        Issue Metadata:
        Title: {issue_metadata['title']}
        Description: {issue_metadata['description']}
        Steps to Reproduce: {issue_metadata['steps_to_reproduce'] if issue_metadata['steps_to_reproduce'] else 'N/A'}
        Expected Behavior: {issue_metadata['expected_behavior'] if issue_metadata['expected_behavior'] else 'N/A'}
        Actual Behavior: {issue_metadata['actual_behavior'] if issue_metadata['actual_behavior'] else 'N/A'}
        Environment: {issue_metadata['environment_details'] if issue_metadata['environment_details'] else 'N/A'}
        Labels: {', '.join(issue_metadata['labels']) if issue_metadata['labels'] else 'N/A'}

        Available Namespaces: {namespace_list}
        
        NAMESPACE EXPLANATION:
        - SOURCECODEFILE:filename.py:ENG - Contains English summaries of functions and API endpoints, choose :ENG when you are focusing on particular function or apiendpoint because the respective context is in english
        - SOURCECODEFILE:filename.py - Contains complete source code of the file. choose if you have work regarding the exact code of the file 
        - TESTINGFILE:filename.py - Contains test files and test cases. Read the file names it defines what kind of test it is
        - DOCUMENTATIONFILE:filename.md - Contains documentation and readme files
        
        OPERATION TYPES:
        - "vector": For semantic similarity searches to find functionally related code, error patterns, 
        behavior analysis, and conceptually similar implementations
        - "graph": For relationship-based searches like function dependencies, class inheritance, 
        method calls, file relationships, and structural code connections
        
        Create 3-4 strategic subqueries that comprehensively investigate different aspects of this issue:

        QUERY CONSTRUCTION GUIDELINES:
        1. VECTOR queries: Use rich, descriptive context with relevant keywords, error messages, 
        function names, and technical details for semantic matching
        2. GRAPH queries: Include specific entity names, relationships, and structural patterns 
        for relationship traversal
        3. Create declarative statements with context (NOT questions)
        4. Include technical terminology, error messages, method names, and domain keywords
        5. Make queries substantial (20-50 words) with sufficient context for effective retrieval
        
        NAMESPACE SELECTION:
        - For understanding functionality/behavior: Use SOURCECODEFILE with :ENG suffix
        - For detailed code analysis/debugging: Use SOURCECODEFILE without :ENG suffix  
        - For test-related investigation: Use TESTINGFILE namespaces
        - For setup/configuration issues: Use DOCUMENTATIONFILE namespaces
        - Specify multiple namespaces if query spans multiple file types
        
        PURPOSE CATEGORIES (choose appropriate purpose for each query):
        - "Root cause identification" - Finding the source of the problem
        - "Functionality analysis" - Understanding how features should work
        - "Error pattern investigation" - Analyzing error messages and failure modes
        - "Dependency analysis" - Understanding code relationships and connections
        - "Test coverage verification" - Checking existing test cases
        - "Configuration analysis" - Examining setup and environment issues
        - "API behavior investigation" - Understanding interface and endpoint behavior
        - "Data flow analysis" - Tracing data movement through the system

        EXAMPLES:
        Vector: "authentication middleware token validation error handling session management invalid credentials response"
        Graph: "UserController authenticate method dependencies database connection service layer relationships"
        
        For each subquery, provide:
        1. query_text: Rich contextual statement with technical details
        2. namespace: List of namespace(s) to search (from available namespaces)
        3. operation_type: "vector" or "graph"
        4. purpose: Clear description of what aspect this query investigates
        
        Format as JSON list:
        [
            {{
                "query_text": "...",
                "namespace": ["SOURCECODEFILE:task_manager.git/task_service.py:py:ENG"],
                "operation_type": "vector",
                "purpose": "..."
            }},
            {{
                "query_text": "...", 
                "namespace": ["TESTINGFILE:task_manager.git/test_api.py:py"],
                "operation_type": "graph",
                "purpose": "..."
            }}
        ]
        
        Ensure your response contains ONLY the JSON list, nothing else.
        """
        
        try:
            parser = JsonOutputParser(pydantic_object=List[SubQuery])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            subqueries = parser.parse(response.content)
            return subqueries
        except Exception as e:
            return [
                SubQuery(
                    query_text=f"Find code related to {issue_metadata['title']}",
                    namespace=None,
                    operation_type="vector",
                    purpose="Root cause identification"
                )
            ]
    
    def process_subquery(self, subquery: SubQuery) -> SearchResult:
        """Process a single subquery using appropriate search method"""
        #logger.info(f"Processing subquery: {subquery['query_text']} using {subquery['operation_type']}")
        
        result = SearchResult(source_type=subquery['operation_type'], insight="")
        
        try:
            if subquery['operation_type'] == "vector":
                # enhanced_query = f"In {subquery['target_scope']}: {subquery['query_text']}"
                for namespace in subquery['namespace']:
                    content = ""
                    code_snippets = pinecone.retrieve_data_from_pincone(context=subquery['query_text'], namespace=namespace)
                    print(f"the code snippets retrieved are {code_snippets}")
                    for data in code_snippets:
                        content += f"{data['content']}\n\n"

                    print(f"the content retrieved is {content}")
                    print(f"The metadata from the retrieval is {code_snippets[0]['metadata']}")

                    vector_insights = self._generate_insight_from_vector(subquery, content, metadata=code_snippets[0]['metadata'])
                    print("\n\n")
                    print(f"the vector insights generated are {vector_insights}")

                    result.code_snippets = vector_insights['relevant_code_sections']
                    result.insight = vector_insights['insights']


            elif subquery['operation_type'] == "graph":
                # Perform graph search
                graph_retrieval = self.run_graph_query(
                    subquery['query_text'],
                    subquery['purpose']
                )
                print('\n\n')
                
                if graph_retrieval['records'] and len(graph_retrieval['records']) > 0:
                    # print(f"the graph retrieval is {graph_retrieval}")
                    graph_insights = self._generate_insight_from_graph(subquery, graph_retrieval)
                    print("\n\n")
                    print(f"the graph insights generated are {graph_insights}")
                    result.insight = graph_insights['insights']
                    result.relationships = graph_insights['relationships']

            print(f"the result generated is {result}")
            #logger.info(f"Completed processing subquery with insight length: {len(result.insight)}")
            return result
            
        except Exception as e:
            return SearchResult(
                source_type=subquery['operation_type'],
                insight=f"Error processing subquery: {str(e)}"
            )
    
    def _generate_insight_from_vector(self, subquery: SubQuery, content, metadata) -> str:
        prompt = f"""
            Analyze the provided file content and metadata to generate actionable insights that help resolve the GitHub issue. Focus on understanding what the code does, how it works, and what changes are needed. Extract specific implementation details, identify root causes for bugs, suggest concrete solutions, and provide clear recommendations for developers.
            Input Structure

            QUERY_TEXT: {subquery['query_text']}
            PURPOSE: {subquery['purpose']}
            CONTENT: {content}
            METADATA: {metadata}    

            File Types and Metadata Information
            Source Code Files
            Metadata provides:

            file_purpose: Overall role of this file in the project architecture
            summary: Detailed explanation of what the file does and its importance
            technologies_descriptions: Tech stack used with explanations
            project_contribution: How this file fits into the broader system
            main_functionality: Core features and capabilities provided
            dependencies: External libraries and modules required
            key_components_descriptions: Important classes, functions, and their roles

            Analysis Focus: Implementation patterns, business logic, data flows, API design, error handling, and integration points.
            Test Files
            Metadata provides:

            file_purpose: Testing scope and objectives
            test_framework_and_approach: Testing methodology and tools used
            test_coverage_summary: What functionality is tested and how
            key_insights: Important testing patterns and considerations

            Analysis Focus: Test coverage gaps, testing strategies, mock data, test scenarios, and quality assurance approach.
            Documentation Files
            Metadata provides:

            doc_purpose: Documentation objective and target audience
            doc_type: Category of documentation (tutorial, API reference, guide, etc.)
            summary: Comprehensive overview of documentation content
            key_concepts: Main topics and concepts covered

            Analysis Focus: Setup instructions, architecture decisions, usage examples, known limitations, and configuration requirements.

            What to Do

            Understand the Context: Use metadata to grasp the file's role and purpose
            Analyze Functionality: Extract what the code actually does and how it works
            Connect to Query: Identify how this file relates to the specific issue or question
            Provide Solutions: Suggest concrete implementation steps or bug fixes
            Assess Impact: Consider testing needs, risks, and broader system effects
            Be Specific: Reference actual code elements, not generic descriptions

            Insights should state exact dependencies, framework and technologies used in the context 

            Return Strict json format JSON format. Explain things inside insights and exact code in relevant_code_sections:

            INSIGHTS SHOULD BE A STRING AND RELEVANT_CODE_SECTIONS SHOULD BE A LIST OF STRING WHICH CONTAINS CODE SNPPETS AT EACH INDEX

            INSIGHTS SHOULD BE DETAILED FORMED USIGN THE CONTEXT AND METADATA 

            json{{
            "insights": "Comprehensive analysis covering: what the code does, how it works, root cause analysis (if bug), implementation approach, specific changes needed, testing strategy, and potential risks. Be specific about class names, method signatures, and code patterns found.",
            "relevant_code_sections": ["Write exact code part"]
            }}

    """
        print("-----------------------VECTOR_INSIGHT-------------------------------")
        parser = JsonOutputParser(pydantic_object=VectorRetrieval)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        # print(response)
        result = parser.parse(response.content)
        print(f"\nthe vector result generated are {result}")
        return result
    
    def _generate_insight_from_graph (self, subquery: SubQuery, graph_retrieval) -> str:
        """Generate insights from search results"""
        print(f"graph data retrived {graph_retrieval['records']}, {graph_retrieval['cypher_query']}")
        prompt = f"""
            Analyze this graph database result for code architecture insights:

            QUERY: {subquery['query_text']}
            PURPOSE: {subquery['purpose']}
            CYPHER: {graph_retrieval['cypher_query']}
            RESULTS: {graph_retrieval['records']}

            Analyze these 5 key areas:
            1. **Relationships**: Dominant types (BELONGS_TO, CALLS, etc.) and patterns
            2. **Dependencies**: Coupling strength, bottlenecks, circular deps
            3. **Architecture**: Component boundaries, layering violations
            4. **Risks**: High-connection nodes, single points of failure
            5. **Organization**: Module cohesion, folder structure effectiveness

            YOU THE SUMMARY AND OTHER PRPOERTY OF PARTICULAR NODE FOR BETTER UNDERSTANDING OF A NODE  

            SHOULD TALK ABOUT THE RELATIONSIPS BETWEEN THE NODES AND THE NODE PROPERTY AND HOW CAN IT HELP TO COMPLETE THE PURPOSE AND SOLVE THE QUERY


            Focus on: structural risks, refactoring opportunities, maintainability impact.
            Be specific and quantitative where possible. 

            Return Strict json format JSON format. Explain things inside insights and exact relationship between codes:
            insights should be long, descriptive and detailed, covering:
            INSIGHTS SHOULD BE A STRING AND RELATIONSHIPS SHOULD BE A LIST OF STRING WHICH 
            INSIGHTS SHOULD BE CREATED WITH THE NODE PROPERTIES.
            DESCRIBE THE RELATIONSHIP BETWEEN THE NODES AND HOW THEY HELP TO SOLVE THE QUERY
            {{
                "insights": "Specific architectural analysis with quantified patterns and actionable recommendations",
                "relationships": ["exact node-to-node relationships found"]
            }}
        """
    
        #logger.debug("Sending insight generation prompt to LLM")
        print("-----------------------XXXXXX_-------------------------------")
        parser = JsonOutputParser(pydantic_object=GraphRetrieval)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        # print(response)
        result = parser.parse(response.content)
        print(f"the graph insight generated: {result}")
        return result
        
    
    # def generate_solution(self, issue_metadata: IssueMetadata, search_results: List[SearchResult]) -> SolutionOutput:
    #     """Generate a comprehensive solution based on all search results"""
    #     #logger.info("Generating solution from search results")
    #     # logger.debug(f"Working with {len(search_results)} search results")
        
    #     # Prepare context from all search results
    #     # results_context = "\n\n".join([
    #     #     f"Subquery Result {i+1}:\n{result.insight}" 
    #     #     for i, result in enumerate(search_results)
    #     # ])

    #     print("entering generate_solution")
    #     print(f"the issue metadata is {issue_metadata}")
    #     print(f"the search results are {search_results}")        
    #     # Prepare code snippets context and extract tech stack
    #     # code_snippets_context = ""
    #     # tech_stack = set()
    #     # for result in search_results:
    #     #     if result.code_snippets:
    #     #         for snippet in result.code_snippets:
    #     #             code_snippets_context += f"\nFile: {snippet.file_path}\n```\n{snippet.code}\n```\n"
    #     #             # Extract tech stack info from metadata
    #     #             if hasattr(snippet, 'metadata'):
    #     #                 if 'language' in snippet.metadata:
    #     #                     tech_stack.add(snippet.metadata['language'])
    #     #                 if 'framework' in snippet.metadata:
    #     #                     tech_stack.add(snippet.metadata['framework'])
    #     #                 if 'library' in snippet.metadata:
    #     #                     tech_stack.add(snippet.metadata['library'])
        
    #     # # Prepare relationships context
    #     # relationships_context = ""
    #     # visualization_query = None
    #     # for result in search_results:
    #     #     if result.relationships and len(result.relationships) > 0:
    #     #         # Use the first relationship's cypher query for visualization if available
    #     #         if not visualization_query and result.relationships[0].cypher_query:
    #     #             visualization_query = result.relationships[0].cypher_query
                
    #     #         for rel in result.relationships:
    #     #             relationships_context += f"\nRelationship: {rel.relationship_type}\n"
    #     #             relationships_context += f"Nodes: {', '.join([str(n) for n in rel.nodes[:2]])}\n"
        
    #     prompt = f"""
    #     Based on the issue and search results, generate a comprehensive solution.
        
    #     Issue:
    #     Title: {issue_metadata['title']}
    #     Description: {issue_metadata['description']}
    #     Expected Behavior: {issue_metadata['expected_behavior'] if issue_metadata['expected_behavior'] else 'N/A'}
    #     Actual Behavior: {issue_metadata['actual_behavior'] if issue_metadata['actual_behavior'] else 'N/A'}
        
    #     Search Results Summary:
    #     {search_results}
        
        
    #     Generate a structured solution with the following components:
        
    #     1. Summary: Provide a brief English summary of the issue and solution (2-3 paragraphs)
    #     2. Procedural Knowledge: Step-by-step instructions for implementing the solution (if applicable)
    #     3. Code Solution: Correct, precise code that addresses the root cause (if applicable)
    #     4. Visualization Query: Cypher query for Neo4j visualization (if applicable)

    #     create visualization query using below given knowledge base structure and using previous generated graph result.
    #     NODES:
    #     - DataFile: id, file_name, file_path, file_ext, summary
    #     - TemplateMarkupFile: id, file_name, file_path, file_ext, summary
    #     - TestingFile: id, file_name, file_path, file_ext, test_framework, test_reference_dict, summary
    #     - DocumentationFile: id, file_name, file_path, file_ext, summary
    #     - APIEndpoint: id, endpoint (URL path), http_method (GET, POST, etc.), summary
    #     - Function: id, function_name, file_path, return_type, summary
    #     - Folder: id, folder_name, directory_path
        
    #     RELATIONSHIPS:
    #     - BELONGS_TO: Connects child nodes to parent nodes (e.g., file->folder, function->file)
    #     - CALLS: Connects functions that call other functions (e.g., functionA->functionB)
    #     - TEST: Connects test files to the functions or API endpoints they test
    #     - All HTTP methods (GET, POST, PUT, DELETE): Connect API routes to API endpoints
        
    #     Your solution should directly address the root cause and take into account the identified tech stack.
        
    #     Return your response in valid JSON format matching the SolutionOutput structure:
    #     {{
    #         "summary": "Brief English summary of the issue and solution",
    #         "procedural_knowledge": ["Step 1...", "Step 2...", ...] or null if not applicable,
    #         "code_solution": "Generated code solution" or null if not applicable,
    #         "visualization_query": "Cypher query for visualization" or null if not applicable
    #     }}
        
    #     Important: Return ONLY the valid JSON object, with no additional text or formatting.
    #     """
        
    #     try:
    #         #logger.debug("Sending solution generation prompt to LLM")
    #         parser = JsonOutputParser(pydantic_object=SolutionOutput)
    #         response = self.llm.invoke([HumanMessage(content=prompt)])
    #         #logger.debug(f"Received solution from LLM: {response.content[:100]}...")
    #         print("the solution generated")
    #         print(response)
            
    #         solution = parser.parse(response.content)
    #         print(json.dumps(solution.dict()))
            
    #         # Add visualization query if available
    #         # if visualization_query and not solution['visualization_query']:
    #         #     solution['visualization_query'] = visualization_query
                
    #         #logger.info(f"Successfully created SolutionOutput with summary length: {len(solution['summary'])}")
    #         return solution
    #     except Exception as e:
    #         logger.error(f"Solution generation error: {str(e)}")
    #         #logger.error(traceback.format_exc())
    #         # Return basic solution if generation fails
    #         return SolutionOutput(
    #             summary=f"Failed to generate complete solution due to error: {str(e)}",
    #             procedural_knowledge=["Not available due to error."],
    #             code_solution=None,
    #             visualization_query=None
    #         )

    def generate_solution(self, issue_metadata: IssueMetadata, search_results: List[SearchResult]) -> SolutionOutput:
        """Generate a comprehensive solution based on all search results"""
        #logger.info("Generating solution from search results")
        # logger.debug(f"Working with {len(search_results)} search results")
        
        print("entering generate_solution")
        # print(f"the issue metadata is {issue_metadata}")
        # print(f"the search results are {search_results}")        

        prompt = f"""
            You are an expert software engineer tasked with analyzing GitHub repository issues and providing comprehensive solutions based on retrieved codebase information.

            ## Issue Details:
            **Title:** {issue_metadata['title']}
            **Description:** {issue_metadata['description']}
            **Expected Behavior:** {issue_metadata['expected_behavior'] if issue_metadata['expected_behavior'] else 'Not specified'}
            **Actual Behavior:** {issue_metadata['actual_behavior'] if issue_metadata['actual_behavior'] else 'Not specified'}

            ## Retrieved Search Results:
            {search_results}

            ### Search Result Analysis Instructions:
            - **Vector Search Results:** Contain code_snippets with contextual code data and insights
            - **Graph Search Results:** Contain relationships showing code structure connections and insights
            - Each result includes an LLM-generated insight explaining the relevance to the issue

            ## Solution Generation Requirements:

            ### 1. Summary (Required)
            Analyze the issue comprehensively and provide a clear, concise explanation in 2-3 paragraphs covering:
            - Root cause identification based on search results
            - Impact assessment on the codebase
            - High-level solution approach

            ### 2. Procedural Knowledge (Optional)
            If the solution requires implementation steps, provide detailed, actionable instructions:
            - Prerequisites and setup requirements
            - Sequential implementation steps
            - Testing and validation procedures
            - Deployment considerations

            ### 3. Code Solution (Optional)
            When code changes are needed, provide:
            - Complete, production-ready code
            - Proper error handling and edge cases
            - Comments explaining critical sections
            - Integration points with existing codebase

            ### 4. Visualization Query (Optional)
            Generate a Cypher query for Neo4j visualization when architectural understanding is beneficial:

            **Available Node Types:**
            - `DataFile`: Properties (id, file_name, file_path, file_ext, summary)
            - `TemplateMarkupFile`: Properties (id, file_name, file_path, file_ext, summary)
            - `TestingFile`: Properties (id, file_name, file_path, file_ext, test_framework, test_reference_dict, summary)
            - `DocumentationFile`: Properties (id, file_name, file_path, file_ext, summary)
            - `APIEndpoint`: Properties (id, endpoint, http_method, summary)
            - `Function`: Properties (id, function_name, file_path, return_type, summary)
            - `Folder`: Properties (id, folder_name, directory_path)

            **Available Relationships:**
            - `BELONGS_TO`: Parent-child relationships (file→folder, function→file)
            - `CALLS`: Function invocation relationships (functionA→functionB)
            - `TEST`: Test coverage relationships (test_file→function/endpoint)
            - `GET`, `POST`, `PUT`, `DELETE`: HTTP method relationships (route→endpoint)

            ## Output Format:
            Return a valid JSON object matching this exact structure:

            ```json
            {{
                "summary": "Comprehensive analysis of the issue and detailed and proposed solution approach",
                "procedural_knowledge": ["Step 1: Detailed instruction", "Step 2: Next action", "..."] or null,
                "code_solution": "Complete code implementation with proper formatting" or null,
                "visualization_query": "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100" or null
            }}
            Critical Instructions:

            Base your solution entirely on the provided search results and issue context
            Ensure all code solutions are syntactically correct and follow best practices
            Make visualization queries specific to the issue's architectural components
            Provide actionable, implementable solutions that directly address the root cause
            Return ONLY the JSON object with no additional text, explanations, or formatting

            Generate the solution now:
        """
        
        try:
            #logger.debug("Sending solution generation prompt to LLM")
            parser = JsonOutputParser(pydantic_object=SolutionOutput)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            #logger.debug(f"Received solution from LLM: {response.content[:100]}...")
            print("the solution generated")
            print(response) 
            
            solution = parser.parse(response.content)
            print(f"the final solution generated is {solution}")
            print(json.dumps(solution))
            
            #logger.info(f"Successfully created SolutionOutput with summary length: {len(solution['summary'])}")
            return solution
        except Exception as e:
            logger.error(f"Solution generation error: {str(e)}")
            #logger.error(traceback.format_exc())
            # Return basic solution if generation fails
            return SolutionOutput(

                summary=f"Failed to generate complete solution due to error: {str(e)}",
                procedural_knowledge=["Not available due to error."],
                code_solution=None,
                visualization_query=None
            )

# ============ LangGraph Flow ============

def build_issue_solver_graph(analyzer: IssueAnalyzer) -> StateGraph:

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
        issue_text = state.get("issue_text", "")
        if not issue_text:
            #logger.error("No issue text provided")
            return {"error": "No issue text provided"}
        
        issue_metadata = analyzer.analyze_issue(issue_text)
        result = {"issue_metadata": issue_metadata}
        return result
    
    def create_subqueries(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for creating subqueries based on issue metadata"""
        #logger.info("CHECKPOINT: Entering create_subqueries node")
        print("Entering create_subqueries")
        
        issue_metadata = IssueAnalyzerState.get_issue_metadata(state)
        if not issue_metadata:
            error_msg = "No issue metadata available"
            #logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        subqueries = analyzer.create_subqueries(issue_metadata)
        print(f"the subqueries created: {subqueries}")
        # Debug the state we're returning
        result = {"subqueries": subqueries}
        #logger.debug(f"Returning from create_subqueries with keys: {list(result.keys())}")
        print(f"create_subqueries returns: {list(result.keys())}")
        
        return result
    
    def process_subqueries(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for processing all subqueries"""
        #logger.info("CHECKPOINT: Entering process_subqueries node")
        print("Entering process_subqueries")
        
        subqueries = IssueAnalyzerState.get_subqueries(state)
        if not subqueries:
            error_msg = "No subqueries available"
            #logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        search_results = []
        for i, subquery in enumerate(subqueries):
            print(f"Processing subquery {i+1}/{len(subqueries)}")
            result = analyzer.process_subquery(subquery)
            # print(f"Subquery {i+1} result: {result}") 
            search_results.append(result)
        
        # Debug the state we're returning
        result = {"search_results": search_results}
        print(f"search_results : {search_results}")
        print(f"process_subqueries returns: {list(result.keys())} with {len(search_results)} results")
        return result
    
    def generate_solution(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for generating the final solution"""
        #logger.info("Entering generate_solution node")
        
        issue_metadata = IssueAnalyzerState.get_issue_metadata(state)
        search_results = IssueAnalyzerState.get_search_results(state)

        # print("Entering generate_solution")
        # print(f"issue_metadata exists: {issue_metadata},\n\n search_results : {search_results}")
        
        if not issue_metadata:
            #logger.error("Missing issue metadata for solution generation")
            return {"error": "Missing required data for solution generation"}
            
        if not search_results:
            #logger.error("Missing search results for solution generation")
            return {"error": "Missing required data for solution generation"}
            
        #logger.debug(f"Issue metadata: {issue_metadata}")
        #logger.debug(f"Search results count: {len(search_results)}")
        
        try:
            #logger.info("Generating solution from analyzer")
            print("Generating solution from analyzer")
            solution = analyzer.generate_solution(issue_metadata, search_results)
            print(f"the solution generated: {solution}")
            #logger.info("Solution generated successfully")
            return {"solution": solution}
        except Exception as e:
            #logger.exception(f"Error generating solution: {str(e)}")
            return {"error": f"Exception in solution generation: {str(e)}"}

    def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for formatting the final output"""
        #logger.info("Entering format_output node")
        
        solution = IssueAnalyzerState.get_solution(state)
        if not solution:
            #logger.error("No solution available for formatting")
            return {"error": "No solution available"}
            
        #logger.debug(f"Solution to format: {solution}")
        
        # Return the state update
        return {"final_output": solution}
    
    # Add nodes to the graph
    #logger.debug("Adding nodes to workflow")
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
    # pinecone_api_key: str = None,
    # pinecone_index: str = None,
    log_level: int = logging.INFO
) -> callable:
    """Create and return the issue solver function with configurable logging"""
    # Setup logging
    #logger.setLevel(log_level)
    #logger.info(f"Creating issue solver with model: {llm_model}")
    
    # Log config info without sensitive data
    #logger.info(f"Neo4j configured: {bool(neo4j_uri and neo4j_username)}")
    # #logger.info(f"Pinecone configured: {bool(pinecone_api_key and pinecone_index)}")
    
    try:
        # Initialize the analyzer
        #logger.info("Initializing issue analyzer")
        analyzer = IssueAnalyzer(
            llm_model=llm_model,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password
            # pinecone_api_key=pinecone_api_key,
            # pinecone_index=pinecone_index
        )
        
        # Build the workflow graph
        #logger.info("Building issue solver workflow")
        workflow = build_issue_solver_graph(analyzer)
        
        # Compile the graph
        #logger.info("Compiling workflow")
        app = workflow.compile()
        #logger.info("Workflow compiled successfully")
        
    except Exception as e:
        logger.exception(f"Error during issue solver initialization: {str(e)}")
        raise
    
    def solve_issue(issue_text: str) -> SolutionOutput:
        """Solve an issue based on the provided description"""
        issue_summary = issue_text[:100] + "..." if len(issue_text) > 100 else issue_text
        #logger.info(f"Processing new issue: {issue_summary}")
        
        try:
            # Run the workflow with the issue text in the initial state
            #logger.info("Invoking workflow")
            start_time = time.time()
            
            initial_state = {"issue_text": issue_text}
            #logger.debug(f"Initial state keys: {list(initial_state.keys())}")
            
            result = app.invoke(initial_state)
            
            # elapsed_time = time.time() - start_time
            #logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")
            #logger.debug(f"Result keys: {list(result.keys())}")
            
            # Extract the solution from the final state
            if "final_output" in result:
                #logger.info("Solution generated successfully")
                return result["final_output"]
            elif "solution" in result:
                #logger.info("Solution found in result")
                return result["solution"]
            elif "error" in result:
                error_msg = f"Error in processing: {result['error']}"
                #logger.error(error_msg)
                return SolutionOutput(
                    summary=error_msg,
                    procedural_knowledge=None,
                    code_solution=None,
                    visualization_query=None
                )
            else:
                error_msg = "Unknown error occurred during processing"
                #logger.error(error_msg)
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

# # Example usage
# if __name__ == "__main__":
#     # Set up environment variables
#     neo4j_uri = os.environ.get("NEO4J_URI")
#     neo4j_username = os.environ.get("NEO4J_USERNAME")
#     neo4j_password = os.environ.get("NEO4J_PASSWORD")
#     # pinecone_api_key = os.environ.get("PINECONE_API_KEY")
#     # pinecone_index = os.environ.get("PINECONE_INDEX")
    
#     # Get log level from environment or default to INFO
#     log_level_name = os.environ.get("LOG_LEVEL", "INFO")
#     log_level = getattr(logging, log_level_name, logging.INFO)
    
#     # Create the issue solver with logging
#     issue_solver = create_issue_solver(
#         neo4j_uri=neo4j_uri,
#         neo4j_username=neo4j_username,
#         neo4j_password=neo4j_password,
#         log_level=log_level
#     )
    
#     # Example issue text
#     example_issue = """
#     Title: API returning 500 error on pagination request
    
#     Description:
#     When trying to paginate through search results using the search API, it works for the first 3 pages but then returns a 500 error on page 4 and beyond.
    
#     Steps to reproduce:
#     1. Call GET /api/search?q=test
#     2. Get the first page of results
#     3. Follow the "next" link 3 times
#     4. Try to access the 4th page
    
#     Expected behavior:
#     All pages should return results with 200 OK
    
#     Actual behavior:
#     Pages 1-3 work fine, but page 4 returns a 500 Internal Server Error
    
#     Error message from logs:
#     "RangeError: Maximum call stack size exceeded at Object.paginate (/src/utils/pagination.js:42:19)"
    
#     Environment:
#     Production server
#     API version: 2.3.1
#     """
    
#     # # Solve the issue with comprehensive logging
#     # #logger.info("Starting issue solving process")
#     # solution = issue_solver(example_issue)
#     # #logger.info("Issue solving completed")
    
#     # # Print the solution
#     # print(json.dumps(solution.dict(), indent=2))
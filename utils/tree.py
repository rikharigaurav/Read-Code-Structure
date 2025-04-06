import subprocess
from utils.pending_rela import pending_rels
import sys
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
import re   
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass
import os
from utils.neodb import app
from dotenv import load_dotenv
from urllib.parse import urlparse
from utils.pinecone_db import pinecone
from utils.memory import analyze_code, generate_file_metadata
from langchain_mistralai import ChatMistralAI


load_dotenv()

@dataclass
class TraversalState:
    node: Node
    parent_id: str
    level: int
    processed: bool = False

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class PathResolver:
    def __init__(self, project_root=os.getenv('projectROOT')):
        self.project_root = Path(project_root).resolve()

    def get_relative_path(self, absolute_path: str) -> str:
        print(f"Debug: absolute_path={absolute_path}")
        print(f"Debug: project_root={self.project_root}")
        
        try:
            abs_path = Path(absolute_path).resolve()
            print(f"Debug: resolved_abs_path={abs_path}")
            
            # Split the path into components
            parts = list(abs_path.parts)
            
            # Find "task_manager.git" in the path
            task_manager_index = -1
            for i, part in enumerate(parts):
                if part == "task_manager.git":
                    task_manager_index = i
                    break
            
            # Take only what's after task_manager.git
            if task_manager_index != -1 and task_manager_index + 1 < len(parts):
                filtered_parts = parts[task_manager_index + 1:]
            else:
                # If we can't find task_manager.git, just use the filename
                filtered_parts = [abs_path.name]
            
            # Remove file extension from last component
            if filtered_parts:
                filename = filtered_parts[-1]
                filtered_parts[-1] = Path(filename).stem
                
            # Join with dots and return
            result = ".".join(filtered_parts)
            print(f"Final relative path: {result}")
            return result
                
        except Exception as e:
            print(f"Unexpected error in get_relative_path: {e}")
            return abs_path.stem  
    # def resolve_import(self, import_path: str, current_file: str) -> str:
    #     current_dir = Path(current_file).parent
    #     return str((current_dir / import_path.replace('.', '/')).with_suffix('.py'))
    
relative_path = PathResolver(os.getenv('projectROOT'))

def extract_imports(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(source_code, "utf8"))

    imports = []
    cursor = tree.walk()

    reached_root = False
    while not reached_root:
        current_node = cursor.node
        if current_node.type in ('import_statement', 'import_from_statement'):
            import_text = current_node.text.decode('utf8')
            imports.append(import_text)

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracing = False

    return imports


async def extract_string_value(node):
    """
    Extract string value from various types of string nodes.
    Handles string literals, f-strings, formatted strings, and concatenated strings.
    
    Args:
        node: A tree-sitter node representing a string or similar expression
        
    Returns:
        str: The extracted string value, or None if extraction fails
    """
    try:
        # Direct string literals
        if node.type in {"string", "string_literal"}:
            # Handle concatenated strings and f-strings
            if node.child_count > 0:
                parts = []
                for child in node.children:
                    if child.type in {"string_content", "escape_sequence", "formatted_string_content"}:
                        parts.append(child.text.decode("utf8"))
                return "".join(parts)
            return node.text.decode("utf8").strip("\"'")
            
        # Handle f-strings and formatted strings
        elif node.type in {"formatted_string", "f_string"}:
            # Extract content from formatted string
            parts = []
            for child in node.children:
                if child.type in {"string_content", "formatted_string_content"}:
                    parts.append(child.text.decode("utf8"))
            return "".join(parts)
            
        # Handle variables that might contain URLs
        elif node.type == "identifier":
            # Just return the variable name for logging purposes
            var_name = node.text.decode("utf8")
            print(f"Variable reference found: {var_name} (cannot extract actual value)")
            return f"VAR:{var_name}"
            
        print(f"Unhandled node type in extract_string_value: {node.type}")
        return None
    except Exception as e:
        print(f"Error extracting string value: {e}")
        return None


def get_function_arguments(function_node):
    """Extract argument names from a function_definition node."""
    parameters_node = next(
        (child for child in function_node.children if child.type == 'parameters'),
        None
    )
    if not parameters_node:
        return []
    
    args = []
    for param in parameters_node.children:
        arg_name = extract_argument_name(param)
        if arg_name:
            args.append(arg_name)
    return args

def extract_argument_name(node):
    """Recursively search for the identifier in a parameter node."""
    if node.type == 'identifier':
        return node.text.decode('utf8')
    for child in node.children:
        result = extract_argument_name(child)
        if result:
            return result
    return None

def get_return_type(node: Node):
    return_type = None
    return_type_node = node.child_by_field_name('return_type')
    if return_type_node:
        return_type = return_type_node.text.decode('utf8').strip()

    return return_type

def get_called_function_name(node: Node) -> str:
    # Get the function name from call expression
    func = node.child_by_field_name('function')
    if func.type == 'identifier':
        return func.text.decode()
    elif func.type == 'attribute':
        return func.child_by_field_name('attribute').text.decode()
    return ''

def install_tree_sitter_language(language: str):
    try:
        package_name = f"tree-sitter-{language}"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
        return package_name.replace("-", "_")
    except subprocess.CalledProcessError:
        try:
            package_name = f"{language}-tree-sitter"
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}")
            return package_name.replace("-", "_")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Failed to install Tree-sitter for language '{language}'. Package not available.")

def get_imports(code: str):
    """
    Extract from-import statements from Python code and return a dictionary
    where keys are imported function names and values are the module paths.
    
    Args:
        code (str): Python source code to analyze
        
    Returns:
        dict: Dictionary mapping function names to their modules
    """
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))

    imports = []
    cursor = tree.walk()

    # Extract import statements
    reached_root = False
    while not reached_root:
        if cursor.node.type == 'import_from_statement':
            imports.append(cursor.node.text.decode('utf8'))

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracing = False

    # Process the imports
    result_dict = {}
    pattern = r'from\s+([\w\.]+)\s+import\s+([\w\s,]+(?:\s+as\s+\w+)?)'

    for imp in imports:
        match = re.match(pattern, imp)
        if match:
            module = match.group(1)  # Module path
            names = match.group(2)   # Imported names

            # Process each imported name
            for item in names.split(','):
                item = item.strip()
                if ' as ' in item:
                    original_name, alias = item.split(' as ', 1)
                    # Store the original name (not the alias)
                    result_dict[original_name.strip()] = module
                else:
                    result_dict[item] = module
    # print(result_dict)
    return result_dict
            
async def read_and_parse(file_path, parent_id, project_root = os.getenv('projectROOT')):
    result = None
    try:
        print("reading parsing is started ----------------------------")
        code = None
        with open(file_path, 'r') as f:
            code = f.read()
        code_bytes = bytes(code, "utf8")
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(code_bytes)

        if project_root:
            project_root = str(Path(file_path).parent)

        path_resolver = PathResolver(project_root)
        imports = get_imports(code)
        initial_state = TraversalState(
            node=tree.root_node,  
            parent_id=parent_id,
            level=1
        )
        allImports = extract_imports(file_path)
        file_Structure = parse_tree(tree.root_node)

        chat = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
        conversation_history = []

        content_string = ""
        content_string += f"#File Structure Summary\n\n-##file path {file_path} \n\n## Code Structure\n\n{result}\n\n## Imports\n{allImports}\n\n##Code Parser \n"
        
        
        updated_content_string, updated_conversation_history = await traverse_tree(
                initial_state, 
                imports, 
                file_path, 
                parent_id, 
                code_bytes, 
                content_string, 
                file_Structure,      
                chat, 
                conversation_history
            )

        file_metadata = generate_file_metadata(
            file_path=file_path,
            file_structure=file_Structure,
            conversation_history=updated_conversation_history,
            llm=chat
        )

        final_content_string = updated_content_string + f"\n\n## File Metadata\n{file_metadata}"
        print("Updated content string with metadata:", final_content_string)

        pinecone.load_text_to_pinecone(
            content=final_content_string, 
            file_id=parent_id, 
            file_structure=file_Structure, 
            metadata = file_metadata
        )
        # print('=============================')
        # print(f"the file structure is {result}")

    except Exception as e:
        print(f"Parsing failed: {str(e)}")

    return file_Structure



async def traverse_tree(
        initial_state: TraversalState, 
        imports: dict, 
        file_path: str, 
        parent_id: str, 
        file_content_bytes: bytes, 
        content_string, 
        file_Structure
    ) -> List[Tuple[str, str, str]]:
    """     
    Perform pre-order DFS traversal using a tree-sitter cursor.
    Returns a list of relationships (source_id, target_id, relationship_type)
    """
    print("traversing is started ----------------------------")

    stack = [initial_state]
    print("🚀 Starting tree traversal...")
    print(f"Initial state: {initial_state}")

    while stack:
        current_state = stack.pop()
        node = current_state.node
        parent_id = current_state.parent_id
        current_level = current_state.level
        
        start_byte = node.start_byte
        end_byte = node.end_byte
        node_text = file_content_bytes[start_byte:end_byte].decode('utf8')
        
        if current_state.processed:
            # print(f"{'  ' * current_level}⏭️  Skipping processed node")
            continue

        if current_level == 1:
            current_level = current_level + 1

        if current_level >= 2:
            if node.type == "call":
                function_node = node.child_by_field_name("function")
                if function_node:
                    print(f"{'  ' * current_level}📞 Processing call node: {function_node.text.decode('utf8')}")
                
                if function_node and function_node.type == "attribute":
                    method_name_node = function_node.child_by_field_name("attribute")
                    object_name_node = function_node.child_by_field_name("object")
                    if method_name_node and object_name_node:
                        # Decode node texts
                        method_name = method_name_node.text.decode("utf8")
                        object_name = object_name_node.text.decode("utf8")
                        
                        if object_name == "requests" and method_name in {"get", "post", "put", "delete", "patch"}:
                            print(f"{'  ' * current_level}🎯 Found method call: {object_name}.{method_name}")
                            argument_node = node.child_by_field_name("arguments")
                            if argument_node and argument_node.child_count > 0:
                                route = None
                                print("Inspecting argument node children:")
                                
                                # Separate positional and keyword arguments
                                positional_args = []
                                keyword_args = {}
                                
                                for child in argument_node.children:
                                    print("Child type:", child.type, "Text:", child.text.decode("utf8"))
                                    if child.type == 'keyword_argument':
                                        key_node = child.child_by_field_name('name')
                                        value_node = child.child_by_field_name('value')
                                        if key_node and value_node:
                                            keyword = key_node.text.decode('utf8')
                                            keyword_args[keyword] = value_node
                                    else:
                                        positional_args.append(child)

                                # Route extraction from arguments
                                route = None
                                url_patterns = ["url", "route", "path", "endpoint"]

                                # First check positional arguments (typically the first argument is the URL)
                                for arg in positional_args:
                                    route = await extract_string_value(arg)
                                    if route:
                                        print(f"Found route in positional argument: {route}")
                                        break

                                # If no route found in positional args, check keyword arguments
                                if not route:
                                    # Check common URL-related keywords
                                    for keyword in url_patterns:
                                        if keyword in keyword_args:
                                            route = await extract_string_value(keyword_args[keyword])
                                            if route:
                                                print(f"Found route in keyword argument '{keyword}': {route}")
                                                break

                                # Process the found route
                                if route:
                                    # Handle variable references specially
                                    if isinstance(route, str) and route.startswith("VAR:"):
                                        var_name = route[4:]  # Extract variable name
                                        print(f"Route is a variable reference: {var_name}, using placeholder")
                                        route = f"VARIABLE_REFERENCE:{var_name}"
                                    else:
                                        # Clean up the route
                                        route = route.strip("\"'")
                                        
                                        # Handle full URLs by extracting just the path
                                        if route.startswith(("http://", "https://", "www.")):
                                            try:
                                                parsed_url = urlparse(route)
                                                route = parsed_url.path or route
                                                print(f"Extracted path from URL: {route}")
                                            except Exception as e:
                                                print(f"Error parsing URL: {e}")
                                else:
                                    print("No route found for API call")
                                    route = "UNKNOWN_ROUTE"  # Fallback value

                                target_id = f"APIENDPOINT:{route}:{method_name.upper()}"
                                if parent_id:
                                    print(f"Detected API Call - method: {method_name.upper()}, route: {route}")
                                    pending_rels.add_relationship(parent_id, target_id, method_name.upper())

                        else:
                            target_id = None
                            for key in imports.keys():
                                if object_name in key:
                                    target_id = f"FUNCTION:{imports[key]}:{method_name}"
                                    break

                            if target_id and parent_id:
                                print(f"----------------------------> relation built {parent_id} --> {target_id}")
                                pending_rels.add_relationship(parent_id, target_id, 'CALLS')

                elif function_node and function_node.type == "identifier":
                        
                        called_func = get_called_function_name(node)
                        # print(f"{'  ' * current_level}📟 Function call: {called_func}")

                        target_id = None
                        for key in imports.keys():
                            if called_func in key:
                                target_id = f"FUNCTION:{imports[key]}:{called_func}"
                                break

                        if target_id and parent_id:
                            print(f"----------------------------> relation built {parent_id} --> {target_id}")
                            pending_rels.add_relationship(parent_id, target_id, 'CALLS')

            elif node.type == "decorated_definition":
                function_node = None
                for child in node.children:
                    if child.type == "function_definition":
                        function_node = child
                        # print(f"{'  ' * current_level}✨ Found decorated function")
                        break

                if not function_node:
                    return content_string

                func_name = None
                if function_node.child_by_field_name("name"):
                    func_name = function_node.child_by_field_name("name").text.decode("utf8")

                # Extract decorators
                decorators = []
                http_method = None
                route = None
                is_api = False

                for child in node.children:
                    if child.type == "decorator":
                        decorator_text = child.text.decode("utf8")
                        decorators.append(decorator_text)
                        print(f"Found decorator: {decorator_text}")

                        # Detect API decorator patterns
                        if "@" in decorator_text and ("route" in decorator_text or any(method in decorator_text for method in ["get", "post", "put", "delete", "patch"])):
                            is_api = True
                            
                            # Extract HTTP method
                            method_match = re.search(r"@(\w+)\.(\w+)\(", decorator_text)
                            if method_match:
                                http_method = method_match.group(2).upper()
                                print(f"Detected HTTP method: {http_method}")
                            else:
                                # Try alternative method detection
                                alt_method_patterns = [
                                    r"methods=\[[\'\"](\w+)[\'\"]", # methods=["GET"]
                                    r"methods=\([\'\"](\w+)[\'\"]", # methods=("GET")
                                    r"method=[\'\"](\w+)[\'\"]"     # method="GET"
                                ]
                                for pattern in alt_method_patterns:
                                    alt_match = re.search(pattern, decorator_text, re.IGNORECASE)
                                    if alt_match:
                                        http_method = alt_match.group(1).upper()
                                        print(f"Detected HTTP method from methods parameter: {http_method}")
                                        break
                                
                                # If still no method, use default based on decorator name
                                if not http_method and "route" in decorator_text:
                                    http_method = "GET"  # Default method for route
                                    print(f"Using default HTTP method: {http_method}")
                            
                            # Try multiple route extraction patterns
                            route_patterns = [
                                r'\("([^"]*)"',           # Double quotes: @app.route("/path")
                                r"\'([^']*)\'",           # Single quotes: @app.route('/path')
                                r'@\w+\.\w+\(([^,\)\'"]+)', # No quotes: @app.route(/path)
                                r'path=[\'\"]([^\'"]+)[\'\"]'  # path parameter: path="/users"
                            ]
                            
                            for pattern in route_patterns:
                                route_match = re.search(pattern, decorator_text)
                                if route_match:
                                    route = route_match.group(1)
                                    print(f"Detected route: {route}")
                                    break

                # Only create API endpoint if both route and http_method are valid
                if is_api and route is not None and http_method is not None:
                    print(f"Creating API endpoint - name: {func_name}, http_method: {http_method}, route: {route}")
                    node_id = f"APIENDPOINT:{route}:{http_method}"
                    print(f"---------------------------->         node created {node_id}")
                    
                    try:
                        app.create_api_endpoint_node(node_id, route, http_method)
                        result = analyze_code(node_text, file_Structure)
                        content_string += f"{node.type}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties} \n route {route} method {http_method}\n\n"
                        
                        # Create relation if parent_id exists
                        if parent_id:
                            app.create_relation(node_id, parent_id, 'BELONGS_TO')
                            
                    except Exception as e:
                        print(f"Error creating API endpoint node: {e}")
                elif is_api:
                    print(f"Warning: Detected API decorator but couldn't extract both route ({route}) and HTTP method ({http_method})")

            elif node.type == 'function_definition':
                func_name = node.child_by_field_name('name').text.decode('utf8')
                params = get_function_arguments(node)
                # params_str = ", ".join(params) if params else ""
                return_type = get_return_type(node)
                relative_file_path = relative_path.get_relative_path(file_path) 
                node_id = f"FUNCTION:{relative_file_path}:{func_name}"
                # print(f"{'  ' * current_level}📝 Processing function definition: {func_name} params: {params} return_type: {return_type} relative_path : {relative_file_path}")
                print(f"NODE ID : {node_id}")
                
                try:
                    app.create_function_node(node_id, func_name, file_path, return_type)
                    result = analyze_code(node_text, file_Structure)
                    content_string += f"{node.type}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties}\n\n"
                    # code: str = node.text.decode('utf8')  
                    # embedding = pinecone.get_embeddings(code)
                    # app.update_node_vector_format(node_id, embedding, 'Function')
                    fixed_parent_id = node_id
                    current_state.processed = True
                    if parent_id:
                        app.create_relation(node_id, parent_id, 'BELONGS_TO')
                    current_state.node_id = node_id
                    current_state.parent_id = fixed_parent_id
                except Exception as e:
                    print(f"Error creating function node: {e}")

        cursor = node.walk()
        current_parent_id = current_state.parent_id
        child_level = current_level
        if cursor.goto_first_child():
            if(current_state.processed == True):
            # current_parent_id = current_node_id # or current_state.parent_id
                child_level = current_level + 1
            # stack.append(current_state)
            while True:
                child_node = cursor.node
                stack.append(TraversalState(child_node, parent_id=current_parent_id, level=child_level, processed=False))
                if not cursor.goto_next_sibling():
                    break
    return content_string

def get_name(node):
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf8')
    return None

def get_body(node):
    for child in node.children:
        if child.type == 'block':
            return child
    return None

def parse_tree(root_node):
    def traverse(current_node):
        current_dict = {}
        if current_node.type in ('class_definition', 'function_definition'):
            name = get_name(current_node)
            body_node = get_body(current_node)
            if body_node:
                
                for statement in body_node.children:
                    child_dict = traverse(statement)
                    current_dict.update(child_dict)
            if name:
                return {(current_node.type, name): current_dict}
            else:
                return {}
        else:
            # Process children for non-class/function nodes
            for child in current_node.children:
                child_dict = traverse(child)
                current_dict.update(child_dict)
            return current_dict
    
    return traverse(root_node)






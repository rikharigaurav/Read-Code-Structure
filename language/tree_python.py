from utils.pending_rela import pending_rels
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
from utils.analyze_sourcecode_file import analyze_code, generate_file_metadata
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
    print(f"Extracting string value from node: {node.type}")
    try:
        extracted_value = None
        
        # Handle f-strings specifically
        if (node.type == "string" and node.text.decode("utf8").startswith('f')):
            print(f"Formatted string found: {node.text.decode('utf8')}")
            full_text = node.text.decode("utf8")

            if full_text.startswith('f"'):
                extracted_value = full_text[2:-1]  
            elif full_text.startswith("f'"):
                extracted_value = full_text[2:-1]  
            else:
                extracted_value = full_text[1:]  
            print(f"F-string content extracted: {extracted_value}")

        # Direct string literals (non f-strings)
        elif node.type in {"string", "string_literal"}:
            print(f"String literal found: {node.text.decode('utf8')}")
            # Handle concatenated strings
            if node.child_count > 0:
                parts = []
                for child in node.children:
                    if child.type in {"string_content", "escape_sequence", "formatted_string_content"}:
                        parts.append(child.text.decode("utf8"))
                extracted_value = "".join(parts)
            else:
                extracted_value = node.text.decode("utf8").strip("\"'")
            
        # Handle variables that might contain URLs
        elif node.type == "identifier":
            print(f"Variable reference found: {node.text.decode('utf8')}")
            var_name = node.text.decode("utf8")
            print(f"Variable reference found: {var_name} (cannot extract actual value)")
            return f"VAR:{var_name}"
        
        # Try to handle parenthesized expressions
        elif node.type == "parenthesized_expression":
            print(f"Parenthesized expression found: {node.text.decode('utf8')}")
            for child in node.children:
                if child.type not in {'(', ')'}:
                    child_value = await extract_string_value(child)
                    if child_value:
                        extracted_value = child_value
                        break
        
        else:
            print(f"Unhandled node type in extract_string_value: {node.type}")
            # Try to get raw text as a last resort
            if hasattr(node, "text"):
                extracted_value = node.text.decode("utf8")
        
        # Process the extracted value if it's a URL
        if extracted_value:
            # Clean up the string (remove quotes if they still exist)
            extracted_value = extracted_value.strip("\"'")
            print(f"Extracted value: {extracted_value}")
            
            # Process URLs to extract and standardize the route
            if extracted_value.startswith(("http://", "https://")):
                try:
                    # For f-strings with variables, we need to handle them differently
                    if '{' in extracted_value and '}' in extracted_value:
                        # This is likely an f-string with variables
                        # print(f"F-string URL with variables detected: {extracted_value}")
                        # Extract the base URL structure and convert variables to placeholder format
                        # Parse the URL but keep the variable placeholders
                        parsed_url = urlparse(extracted_value.split('{')[0])  # Parse up to first variable
                        base_path = parsed_url.path
                        
                        # Reconstruct the full path with variables in the correct format
                        # Convert {variable} to <variable> for standardization
                        standardized_path = re.sub(r'\{([^}]+)\}', r'<\1>', extracted_value)
                        
                        # Extract just the path portion if it's a full URL
                        if standardized_path.startswith(("http://", "https://")):
                            try:
                                # For URLs with variables, we need to be more careful
                                # Split by '/' and reconstruct the path
                                url_parts = extracted_value.split('/')
                                path_parts = []
                                found_domain = False
                                for part in url_parts:
                                    if found_domain:
                                        path_parts.append(part)
                                    elif '.' in part or part.startswith('localhost'):
                                        found_domain = True
                                
                                if path_parts:
                                    full_path = '/' + '/'.join(path_parts)
                                    standardized_path = re.sub(r'\{([^}]+)\}', r'<\1>', full_path)
                                else:
                                    # Fallback to original extraction method
                                    parsed_url = urlparse(extracted_value.replace('{', '%7B').replace('}', '%7D'))
                                    standardized_path = parsed_url.path.replace('%7B', '<').replace('%7D', '>')
                                    
                            except Exception as e:
                                print(f"Error parsing URL with variables: {e}")
                                # Fallback: just extract path-like portion
                                standardized_path = re.sub(r'https?://[^/]+', '', extracted_value)
                                standardized_path = re.sub(r'\{([^}]+)\}', r'<\1>', standardized_path)
                        
                        # print(f"Extracted standardized path from f-string URL: {standardized_path}")
                        return standardized_path
                    else:
                        # Regular URL without variables
                        parsed_url = urlparse(extracted_value)
                        path = parsed_url.path
                        # print(f"Parsed URL: {parsed_url}")
                        # print(f"Extracted path: {path}")
                        standardized_path = re.sub(r'\{([^}]+)\}', r'<\1>', path)
                        # print(f"Extracted standardized path from URL: {standardized_path}")
                        return standardized_path
                        
                except Exception as e:
                    print(f"Error processing URL: {e}")
                    return extracted_value
            
            return extracted_value
        
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
            
async def read_and_parse_python(file_path, parent_id, project_root = os.getenv('projectROOT')):
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
        content_string += f"#File Structure Summary\n\n-##file path {file_path} \n\n## Code Structure\n\n{file_Structure}\n\n## Imports\n{allImports}\n\n##Code Parser \n"
        
        
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

        # updated_content_string = updated_content_string + f"\n\n## File Metadata\n{file_metadata}"
        print("Updated content string with metadata:", updated_content_string)
        file_metadata['file_structure'] = f"{file_Structure}"
        print(f"the file structure is {file_Structure}")

        pinecone.load_text_to_pinecone(
            content=updated_content_string, 
            file_id=parent_id, 
            metadata = file_metadata
        )

    except Exception as e:
        print(f"Parsing failed: {str(e)}")

    return file_metadata

async def traverse_tree(
        initial_state: TraversalState, 
        imports: dict, 
        file_path: str, 
        parent_id: str, 
        file_content_bytes: bytes, 
        content_string, 
        file_Structure,
        chat, 
        conversation_history
    ) -> List[Tuple[str, str, str]]:
    print("traversing is started ----------------------------")

    stack = [initial_state]
    print("🚀 Starting tree traversal...")
    print(f"Initial state: {initial_state}")
    
    # Track processed API endpoints to avoid duplicate function nodes
    processed_api_functions = set()

    while stack:
        current_state = stack.pop()
        node = current_state.node
        parent_id = current_state.parent_id
        current_level = current_state.level
        
        start_byte = node.start_byte
        end_byte = node.end_byte
        node_text = file_content_bytes[start_byte:end_byte].decode('utf8')
        
        if current_state.processed:
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
                        
                        # Improved API call detection
                        is_api_call = False
                        if object_name == "requests" and method_name in {"get", "post", "put", "delete", "patch"}:
                            is_api_call = True
                        elif object_name.lower() in {"client", "api", "http", "session"} and method_name in {"get", "post", "put", "delete", "patch", "request"}:
                            is_api_call = True
                        
                        if is_api_call:
                            print(f"{'  ' * current_level}🎯 Found API call: {object_name}.{method_name}")
                            argument_node = node.child_by_field_name("arguments")
                            print(f"Argument node: {argument_node}")
                            if argument_node and argument_node.child_count > 0:
                                route = None
                                print("Inspecting argument node children:")
                                
                                positional_args = []
                                keyword_args = {}
                                
                                for child in argument_node.children:
                                    if child.type == 'comma':
                                        continue  # Skip commas
                                        
                                    print("Child type:", child.type, "Text:", child.text.decode("utf8"))
                                    if child.type == 'keyword_argument':
                                        key_node = child.child_by_field_name('name')
                                        value_node = child.child_by_field_name('value')
                                        if key_node and value_node:
                                            keyword = key_node.text.decode('utf8')
                                            keyword_args[keyword] = value_node
                                    elif child.type not in {'comment', 'parenthesized_expression', '(', ')'}:
                                        positional_args.append(child)

                                # Route extraction from arguments
                                route = None
                                url_patterns = ["url", "route", "path", "endpoint", "uri"]

                                # First check positional arguments (typically the first argument is the URL)
                                if positional_args:
                                    route = await extract_string_value(positional_args[0])
                                    if route:
                                        print(f"Found route in first positional argument: {route}")

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

            elif node.type == 'function_definition':
                func_name = node.child_by_field_name('name').text.decode('utf8')
                
                # Skip if this function was already processed as an API endpoint
                if func_name in processed_api_functions:
                    print(f"Skipping function node creation for {func_name} as it was already processed as API endpoint")
                    continue
                
                # params = get_function_arguments(node)
                return_type = get_return_type(node)
                relative_file_path = relative_path.get_relative_path(file_path) 
                node_id = f"FUNCTION:{relative_file_path}:{func_name}"
                print(f"NODE ID : {node_id}")
                
                try:
                    result = analyze_code(node_text, file_Structure)
                    content_string += f"{func_name}:{node.type}\nCode:\n{node_text}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties}\n\n"
                    conversation_history.append({
                        "name": func_name,
                        "type": "function",
                        "purpose": result.purpose,
                        "summary": result.intuition
                    })
                    app.create_function_node(node_id, func_name, file_path, return_type, summary=result.intuition)
                    fixed_parent_id = node_id
                    current_state.processed = True
                    if parent_id:
                        app.create_relation(node_id, parent_id, 'BELONGS_TO')
                    current_state.node_id = node_id
                    current_state.parent_id = fixed_parent_id
                except Exception as e:
                    print(f"Error creating function node: {e}")


            elif node.type == "decorated_definition":
                function_node = None
                for child in node.children:
                    if child.type == "function_definition":
                        function_node = child
                        break

                if not function_node:
                    continue

                func_name = None
                if function_node.child_by_field_name("name"):
                    func_name = function_node.child_by_field_name("name").text.decode("utf8")

                decorators = []
                http_method = None
                route = None
                is_api = False
                api_node_id = None

                for child in node.children:
                    if child.type == "decorator":
                        decorator_text = child.text.decode("utf8")
                        decorators.append(decorator_text)
                        print(f"Found decorator: {decorator_text}")

                        api_decorator_patterns = [
                            r"@\w+\.route", 
                            r"@\w+\.(get|post|put|delete|patch)",
                            r"@\w+\.api\.",
                            r"@\w+\.blueprint\.",
                            r"@\w+_bp\.",
                            r"@\w+_blueprint\.",
                            r"@router\.",
                            r"@api\.",
                            r"@app\."
                        ]
                        
                        is_api_decorator = any(re.search(pattern, decorator_text, re.IGNORECASE) for pattern in api_decorator_patterns)
                        
                        if is_api_decorator:
                            is_api = True
                            
                            method_match = re.search(r"@(\w+)\.(\w+)\(", decorator_text)
                            if method_match:
                                http_method = method_match.group(2).upper()
                                print(f"Detected HTTP method: {http_method}")
                                
                                # Handle route methods directly
                                if http_method.upper() == "ROUTE":
                                    # Look for methods parameter
                                    methods_match = re.search(r"methods=\[?[\'\"](\w+)[\'\"]", decorator_text)
                                    if methods_match:
                                        http_method = methods_match.group(1).upper()
                                    else:
                                        http_method = "GET"  # Default for route
                            else:
                                # Try alternative method detection
                                alt_method_patterns = [
                                    r"methods=\[[\'\"](\w+)[\'\"]", # methods=["GET"]
                                    r"methods=\([\'\"](\w+)[\'\"]", # methods=("GET")
                                    r"method=[\'\"](\w+)[\'\"]",     # method="GET"
                                    r"methods=\[([^\]]*)\]"  # methods=[...]
                                ]
                                for pattern in alt_method_patterns:
                                    alt_match = re.search(pattern, decorator_text, re.IGNORECASE)
                                    if alt_match:
                                        methods_str = alt_match.group(1)
                                        # Extract first method if multiple
                                        if "," in methods_str:
                                            http_method = methods_str.split(",")[0].strip("'\" ").upper()
                                        else:
                                            http_method = methods_str.strip("'\" ").upper()
                                        print(f"Detected HTTP method from methods parameter: {http_method}")
                                        break
                                
                                # If still no method, use default based on decorator name
                                if not http_method:
                                    for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                                        if method.lower() in decorator_text.lower():
                                            http_method = method
                                            print(f"Detected HTTP method from decorator name: {http_method}")
                                            break
                                    else:
                                        http_method = "GET"  # Default method for route
                                        print(f"Using default HTTP method: {http_method}")
                            
                            # Try multiple route extraction patterns
                            route_patterns = [
                                r'\("([^"]*)"',           # Double quotes: @app.route("/path")
                                r"\'([^']*)\'",           # Single quotes: @app.route('/path')
                                r'@\w+\.\w+\(([^,\)\'"]+)', # No quotes: @app.route(/path)
                                r'path=[\'\"]([^\'"]+)[\'\"]',  # path parameter: path="/users"
                                r'prefix=[\'\"]([^\'"]+)[\'\"]'  # prefix parameter: prefix="/api/v1"
                            ]
                            
                            for pattern in route_patterns:
                                route_match = re.search(pattern, decorator_text)
                                if route_match:
                                    route = route_match.group(1)
                                    print(f"Detected route: {route}")
                                    break

                # Create API endpoint if detected
                if is_api and route is not None and http_method is not None and route != 'UNKNOWN_ROUTE':
                    print(f"Creating API endpoint - name: {func_name}, http_method: {http_method}, route: {route}")
                    api_node_id = f"APIENDPOINT:{route}:{http_method}"
                    print(f"---------------------------->         node created {api_node_id}")
                    
                    try:
                        result = analyze_code(node_text, file_Structure)
                        content_string += f"API Endpoint\nCode:\n{node_text}\nRoute: {route}\nMethod: {http_method}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties}\n\n"

                        conversation_history.append({
                            "name": route,
                            "type": "api_endpoint",
                            "purpose": result.purpose,
                            "summary": result.intuition
                        })
                        app.create_api_endpoint_node(api_node_id, route, http_method, summary=result.intuition)
                        #
                        if parent_id:
                            app.create_relation(api_node_id, parent_id, 'BELONGS_TO')
                            
                        # Mark this function as processed as API endpoint
                        processed_api_functions.add(func_name)
                        
                    except Exception as e:
                        print(f"Error creating API endpoint node: {e}")
                elif is_api:
                    print(f"Warning: Detected API decorator but couldn't extract both route ({route}) and HTTP method ({http_method})")

            

        cursor = node.walk()
        current_parent_id = current_state.parent_id
        child_level = current_level
        if cursor.goto_first_child():
            if(current_state.processed == True):
                child_level = current_level + 1
            while True:
                child_node = cursor.node
                stack.append(TraversalState(child_node, parent_id=current_parent_id, level=child_level, processed=False))
                if not cursor.goto_next_sibling():
                    break
    return content_string, conversation_history

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

import re

def parse_tree(root_node):
    def is_api_endpoint(node):
        """Check if a function node is an API endpoint by examining its decorators"""
        # If this is a decorated_definition, check the decorators
        if node.type == "decorated_definition":
            decorators = []
            for child in node.children:
                if child.type == "decorator":
                    decorator_text = child.text.decode("utf8")
                    decorators.append(decorator_text)
            
            # Check if any decorator indicates this is an API endpoint
            api_decorator_patterns = [
                r"@\w+\.route", 
                r"@\w+\.(get|post|put|delete|patch)",
                r"@\w+\.api\.",
                r"@\w+\.blueprint\.",
                r"@\w+_bp\.",
                r"@\w+_blueprint\.",
                r"@router\.",
                r"@api\.",
                r"@app\."
            ]
            
            for decorator in decorators:
                if any(re.search(pattern, decorator, re.IGNORECASE) for pattern in api_decorator_patterns):
                    return True
        
        return False
    
    def get_api_info(node):
        """Extract HTTP method and route from API endpoint decorators"""
        http_method = None
        route = None
        
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    decorator_text = child.text.decode("utf8")
                    
                    method_match = re.search(r"@(\w+)\.(\w+)\(", decorator_text)
                    if method_match:
                        http_method = method_match.group(2).upper()
                        
                        if http_method.upper() == "ROUTE":
                            methods_match = re.search(r"methods=\[?[\'\"](\w+)[\'\"]", decorator_text)
                            if methods_match:
                                http_method = methods_match.group(1).upper()
                            else:
                                http_method = "GET" 
                    
                    if not http_method:
                        alt_method_patterns = [
                            r"methods=\[[\'\"](\w+)[\'\"]",
                            r"methods=\([\'\"](\w+)[\'\"]",
                            r"method=[\'\"](\w+)[\'\"]"
                        ]
                        for pattern in alt_method_patterns:
                            alt_match = re.search(pattern, decorator_text, re.IGNORECASE)
                            if alt_match:
                                http_method = alt_match.group(1).upper()
                                break
                        
                        if not http_method:
                            for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                                if method.lower() in decorator_text.lower():
                                    http_method = method
                                    break
                            else:
                                http_method = "GET"  
                    route_patterns = [
                        r'\("([^"]*)"',           
                        r"\'([^']*)\'",           
                        r'@\w+\.\w+\(([^,\)\'"]+)', 
                        r'path=[\'\"]([^\'"]+)[\'\"]', 
                        r'prefix=[\'\"]([^\'"]+)[\'\"]' 
                    ]
                    
                    for pattern in route_patterns:
                        route_match = re.search(pattern, decorator_text)
                        if route_match:
                            route = route_match.group(1)
                            break
        
        return http_method, route
    
    def get_function_name(node):
        """Extract function name from function_definition or decorated_definition"""
        if node.type == "function_definition":
            name_node = node.child_by_field_name('name')
            if name_node:
                return name_node.text.decode('utf8')
        elif node.type == "decorated_definition":
            # Find the function_definition child
            for child in node.children:
                if child.type == "function_definition":
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        return name_node.text.decode('utf8')
        return None
    
    def get_name(current_node):
        """Get name for class or function definitions"""
        if current_node.type == 'class_definition':
            name_node = current_node.child_by_field_name('name')
            if name_node:
                return name_node.text.decode('utf8')
        elif current_node.type in ('function_definition', 'decorated_definition'):
            return get_function_name(current_node)
        return None
    
    def get_body(current_node):
        """Get body node for class or function definitions"""
        if current_node.type == 'class_definition':
            return current_node.child_by_field_name('body')
        elif current_node.type == 'function_definition':
            return current_node.child_by_field_name('body')
        elif current_node.type == 'decorated_definition':
            # Find the function_definition child and get its body
            for child in current_node.children:
                if child.type == "function_definition":
                    return child.child_by_field_name('body')
        return None

    def traverse(current_node):
        current_dict = {}
        
        # Handle class definitions
        if current_node.type == 'class_definition':
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
        
        # Handle function definitions and decorated definitions
        elif current_node.type in ('function_definition', 'decorated_definition'):
            name = get_name(current_node)
            
            # Check if this is an API endpoint
            if is_api_endpoint(current_node):
                http_method, route = get_api_info(current_node)
                if route and http_method:
                    # This is an API endpoint
                    body_node = get_body(current_node)
                    if body_node:
                        for statement in body_node.children:
                            child_dict = traverse(statement)
                            current_dict.update(child_dict)
                    
                    # Create API endpoint key with route and method
                    api_key = f"{route}:{http_method}"
                    return {('api_endpoint', api_key): current_dict}
            
            # This is a regular function
            body_node = get_body(current_node)
            if body_node:
                for statement in body_node.children:
                    child_dict = traverse(statement)
                    current_dict.update(child_dict)
            if name:
                return {('function_definition', name): current_dict}
            else:
                return {}
        
        else:
            # Process children for non-class/function nodes
            for child in current_node.children:
                child_dict = traverse(child)
                current_dict.update(child_dict)
            return current_dict
    
    return traverse(root_node)





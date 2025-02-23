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
        abs_path = Path(absolute_path).resolve()
        try:
            # Get relative path to project root
            rel_path = abs_path.relative_to(self.project_root)
            
            # Convert path components to list
            parts = list(rel_path.parts)
            
            if not parts:
                return ""
            
            # Remove "Read-Code-Structure" from parts if it exists
            if parts and parts[0] == "Read-Code-Structure":
                parts = parts[1:]
                
            # If no parts left after removal
            if not parts:
                return ""
                
            # Remove file extension from last component
            filename = parts[-1]
            parts[-1] = Path(filename).stem
            
            # Join with dots and return
            return ".".join(parts)
            
        except ValueError:
            return ""  # Path is outside project root

    # def resolve_import(self, import_path: str, current_file: str) -> str:
    #     current_dir = Path(current_file).parent
    #     return str((current_dir / import_path.replace('.', '/')).with_suffix('.py'))
    
relative_path = PathResolver(os.getenv('projectROOT'))

def extract_api_route(node, current_level=0):
    """
    Extract API route from a request method call node.
    Returns tuple of (route, method_name)
    """
    if not node:
        return None, None

    # Get the method name (get, post, etc)
    method_node = node.child_by_field_name("method")
    if not method_node:
        return None, None
    
    method_name = method_node.text.decode("utf8")
    
    # Get arguments node
    argument_node = node.child_by_field_name("arguments")
    if not argument_node:
        return None, method_name

    # Extract route from first string argument
    route = None
    for child in argument_node.named_children:
        # Handle different types of string arguments
        if child.type in {"string", "string_literal", "formatted_string"}:
            if child.type == "formatted_string":
                # Handle f-strings
                parts = []
                for part in child.named_children:
                    if part.type == "string_content":
                        parts.append(part.text.decode("utf8"))
                    elif part.type == "interpolation":
                        parts.append("{}")  # Placeholder for interpolated values
                route = "".join(parts)
            else:
                # Handle regular strings
                route = child.text.decode("utf8").strip("\"'")
            break
        elif child.type == "call":
            # Handle cases where the URL is constructed by a function call
            route = "DYNAMIC_ROUTE"
            break
        elif child.type == "identifier":
            # Handle cases where the URL is stored in a variable
            route = f"VAR_{child.text.decode('utf8')}"
            break

    return route, method_name

def process_request_call(node, current_level, parent_id, pending_rels):
    """
    Process a request method call and create appropriate relationships.
    """
    route, method_name = extract_api_route(node, current_level)
    
    print(f"{'  ' * current_level}🎯 Processing API call:")
    print(f"{'  ' * (current_level + 1)}Method: {method_name}")
    print(f"{'  ' * (current_level + 1)}Route: {route}")

    if route:
        method_upper = method_name.upper()
        target_id = f"APIENDPOINT:{route}:{method_upper}"
        
        if parent_id:
            print(f"{'  ' * current_level}Creating relationship:")
            print(f"{'  ' * (current_level + 1)}From: {parent_id}")
            print(f"{'  ' * (current_level + 1)}To: {target_id}")
            print(f"{'  ' * (current_level + 1)}Type: {method_upper}")
            
            pending_rels.add_relationship(parent_id, target_id, method_upper)
            return True
    
    return False

#-----------------------------------------

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

#Returns the import part of a code file

def get_imports(code: str):
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))

    imports = []
    cursor = tree.walk()

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

    import_dict = {}

    pattern = r'from\s+([\w\.]+)\s+import\s+([\w\s,]+(?:\s+as\s+\w+)?)'

    for imp in imports:
        match = re.match(pattern, imp)
        if match:
            module = match.group(1)  # Extract module name
            names = match.group(2)  # Extract function(s)

            function_names = []
            for item in names.split(','):
                parts = item.strip().split(' as ')  # Handle aliasing
                if len(parts) == 2:
                    function_names.extend(parts[::-1])  # Store alias first, original second
                else:
                    function_names.append(parts[0])

            import_dict[tuple(function_names)] = module

    # return import_dict  # Convert tuple keys to lists
    list_like_dict = {}
    for key, value in import_dict.items():
        list_like_key = str(list(key))  # Convert list to string representation
        list_like_dict[list_like_key] = value

    return list_like_dict
            
async def read_and_parse(file_path, parent_id,  project_root = os.getenv('projectROOT')):
    try:
        code = None
        with open(file_path, 'r') as f:
            code = f.read()
            print("----------------> CODE ")
            # print(code)

        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(bytes(code, "utf8"))

        if tree :
            print("--------------> TREE")
        if project_root:
            project_root = str(Path(file_path).parent)

        path_resolver = PathResolver(project_root)
        imports = get_imports(code)
        print("IMPORTD")
        print(imports)
        initial_state = TraversalState(
            node=tree.root_node,  
            parent_id=parent_id,
            level=1
        )
        print(f"---------initial state-----------> {initial_state}")
        await traverse_tree(initial_state, imports, file_path, path_resolver, parent_id)

    except Exception as e:
        print(f"Parsing failed: {str(e)}")

    return True
        


async def traverse_tree(initial_state: TraversalState, imports: dict, file_path: str, pathResolver: PathResolver, parent_id: str) -> List[Tuple[str, str, str]]:
    """
    Perform pre-order DFS traversal using a tree-sitter cursor.
    Returns a list of relationships (source_id, target_id, relationship_type)
    """
    
    def debug_print_node(node, level, msg_type="INFO"):
        indent = "  " * level
        if node.type == "call":
            print(f"{indent}🔵 [{msg_type}] Call Node: {node.text.decode('utf8')} (Line: {node.start_point[0] + 1})")
        elif node.type == "function_definition":
            print(f"{indent}🟢 [{msg_type}] Function Definition: {node.text.decode('utf8').split('(')[0]} (Line: {node.start_point[0] + 1})")
        elif node.type == "decorated_definition":
            print(f"{indent}🟡 [{msg_type}] Decorated Definition (Line: {node.start_point[0] + 1})")
        elif node.type == "decorator":
            print(f"{indent}🟣 [{msg_type}] Decorator: {node.text.decode('utf8')} (Line: {node.start_point[0] + 1})")
        elif node.type == "attribute":
            print(f"{indent}🟤 [{msg_type}] Attribute Access: {node.text.decode('utf8')} (Line: {node.start_point[0] + 1})")
        else:
            print(f"{indent}⚪ [{msg_type}] {node.type}: {node.text.decode('utf8')} (Line: {node.start_point[0] + 1})")

    stack = [initial_state]
    print("🚀 Starting tree traversal...")
    print(f"Initial state: {initial_state}")

    while stack:
        current_state = stack.pop()
        node = current_state.node
        parent_id = current_state.parent_id
        current_level = current_state.level

        # debug_print_node(node, current_level)
        # print(f"{'  ' * current_level}📍 Current state: level={current_level}, processed={current_state.processed}, parent_id={parent_id}")

        if current_state.processed:
            # print(f"{'  ' * current_level}⏭️  Skipping processed node")
            continue

        if current_level == 1:
            current_level = current_level + 1

        if current_level >= 2:
            if node.type == "call":
                function_node = node.child_by_field_name("function")
                # if function_node:
                    # print(f"{'  ' * current_level}📞 Processing call node: {function_node.text.decode('utf8')}")
                
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
                                for child in argument_node.children:
                                    print("Child type:", child.type, "Text:", child.text.decode("utf8"))
                                    if child.type in {"string", "string_literal", "string_content"}:
                                        # Handle compound string nodes
                                        if child.child_count > 0:
                                            route = "".join(c.text.decode("utf8") for c in child.children)
                                        else:
                                            route = child.text.decode("utf8")
                                        break
                                if route is not None:
                                    route = route.strip("\"'")
                                    # Extract only the path from the URL
                                    parsed_url = urlparse(route)
                                    if parsed_url.path:
                                        route = parsed_url.path
                                    print("Extracted route:", route)
                                else:
                                    print("No literal route found for API call")

                                target_id = f"APIENDPOINT:{route}:{method_name.upper()}"
                                                                
                                if target_id and parent_id:
                                    print(f"Detected API Call - method: {method_name.upper()}, route: {route}")
                                    pending_rels.add_relationship(parent_id, target_id, method_name.upper())


                elif function_node and function_node.type == "identifier":
                        
                        called_func = get_called_function_name(node)
                        print(f"{'  ' * current_level}📟 Function call: {called_func}")

                        target_id = None
                        for key in imports.keys():
                            if called_func in key:
                                target_id = f"FUNCTION:{imports[key]}:{called_func}"
                                break

                        if target_id and parent_id:
                            print(f"----------------------------> relation built {parent_id} --> {target_id}")
                            pending_rels.add_relationship(parent_id, target_id, 'CALLS')

            if node.type == "decorated_definition":
                # print(f"{'  ' * current_level}🎭 Processing decorated definition")
                function_node = None
                for child in node.children:
                    if child.type == "function_definition":
                        function_node = child
                        # print(f"{'  ' * current_level}✨ Found decorated function")
                        break

                if not function_node:
                    return  # Skip if no function is found

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

                        # Detect API decorator patterns
                        if "@" in decorator_text and ("route" in decorator_text or "get" in decorator_text or "post" in decorator_text):
                            is_api = True
                            method_match = re.search(r"@(\w+)\.(\w+)\(", decorator_text)
                            if method_match:
                                http_method = method_match.group(2).upper()
                            route_match = re.search(r'\("(.*?)"', decorator_text)
                            if route_match:
                                route = route_match.group(1)

                # Store results
                if is_api:
                    # print(f"name: {func_name} http_method: {http_method}, route: {route}, decorators: {decorators}")
                    node_id = f"APIENDPOINT:{route}:{http_method}"
                    # print(f"---------------------------->         node created {node_id}")
                    app.create_api_endpoint_node(node_id, route, http_method)
                    if parent_id:
                        # print(f"---------------------------->         relation built {node_id}  --> {current_state.parent_id}")
                        app.create_relation(node_id, parent_id, 'BELONGS_TO')


            elif node.type == 'function_definition':
                func_name = node.child_by_field_name('name').text.decode('utf8')
                params = get_function_arguments(node)
                # params_str = ", ".join(params) if params else ""
                return_type = get_return_type(node)
                relative_file_path = relative_path.get_relative_path(file_path) 
                node_id = f"FUNCTION:{relative_file_path}:{func_name}"
                # print(f"{'  ' * current_level}📝 Processing function definition: {func_name} params: {params} return_type: {return_type} relative_path : {relative_file_path}")
                # print(f"NODE ID : {node_id}")
                app.create_function_node(node_id, func_name, file_path, return_type)
                # print(f"---------------------------->         node created {node_id}")
                fixed_parent_id = node_id
                current_state.processed = True
                if parent_id:
                    app.create_relation(node_id, parent_id, 'BELONGS_TO')
                current_state.node_id = node_id
                current_state.parent_id = fixed_parent_id


        
        # Push children onto the stack in reverse order to maintain pre-order traversal.
        cursor = node.walk()
        current_parent_id = current_state.parent_id
        child_level = current_level
        if cursor.goto_first_child():
            if(current_state.processed == True):
            # current_parent_id = current_node_id # or current_state.parent_id
                child_level = current_level + 1
            # stack.append(current_state)
            # Iterate over all children
            while True:
                child_node = cursor.node
                stack.append(TraversalState(child_node, parent_id=current_parent_id, level=child_level, processed=False))
                if not cursor.goto_next_sibling():
                    break






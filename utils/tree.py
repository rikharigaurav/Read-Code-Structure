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
from utils.memory import analyze_code


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
    if node.type in {"string", "string_literal"}:
        # Handle concatenated strings and f-strings
        if node.child_count > 0:
            return "".join([
                c.text.decode("utf8") for c in node.children
                if c.type in {"string_content", "escape_sequence"}
            ])
        return node.text.decode("utf8").strip("\"'")
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
            
async def read_and_parse(file_path, parent_id,  project_root = os.getenv('projectROOT')):
    result = None
    try:
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
        content_string += f"#File Structure Summary\n\n-##file path {file_path} \n\n## Code Structure\n\n{result}\n\n## Imports\n{allImports}\n\n##Code Parser \n"
        updated_content_string = await traverse_tree(initial_state, imports, file_path, parent_id, code_bytes, content_string, file_Structure)
        print("Updated content string:", updated_content_string)
        pinecone.load_text_to_pinecone(content=updated_content_string, file_id=parent_id, file_structure=file_Structure)
        # print('=============================')
        # print(f"the file structure is {result}")

    except Exception as e:
        print(f"Parsing failed: {str(e)}")

    return file_Structure
        


async def traverse_tree(initial_state: TraversalState, imports: dict, file_path: str, parent_id: str, file_content_bytes: bytes, content_string, file_Structure) -> List[Tuple[str, str, str]]:
    """     
    Perform pre-order DFS traversal using a tree-sitter cursor.
    Returns a list of relationships (source_id, target_id, relationship_type)
    """


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

                                # Check positional arguments first
                                for arg in positional_args:
                                    route = await extract_string_value(arg)
                                    if route: 
                                        break

                                # If no route found, check url keyword
                                if not route and 'url' in keyword_args:
                                    route = await extract_string_value(keyword_args['url'])

                                # Process found route
                                if route:
                                    route = route.strip("\"'")
                                    parsed_url = urlparse(route)
                                    route = parsed_url.path or route 
                                    print("Extracted route:", route)
                                else:
                                    print("No literal route found for API call")
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

                    result = await analyze_code(node_text, file_Structure)
                    content_string += f"{node.type}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties} \n route {route} method {http_method}\n\n"
                    # code: str = node.text.decode('utf8')  
                    # embedding =  pinecone.get_embeddings(code) 
                    # app.update_node_vector_format(node_id, embedding, 'APIEndpoint')
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
                result = await analyze_code(node_text, file_Structure)
                content_string += f"{node.type}\nPurpose: {result.purpose}\nIntuition: {result.intuition}\n{result.properties}"
                # code: str = node.text.decode('utf8')  
                # embedding = pinecone.get_embeddings(code)
                # app.update_node_vector_format(node_id, embedding, 'Function')
                fixed_parent_id = node_id
                current_state.processed = True
                if parent_id:
                    app.create_relation(node_id, parent_id, 'BELONGS_TO')
                current_state.node_id = node_id
                current_state.parent_id = fixed_parent_id

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






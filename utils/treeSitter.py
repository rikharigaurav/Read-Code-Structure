import subprocess
from utils.langgraph import pending_rels
import sys
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
import re
from typing import List, Dict, TypedDict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import os
from utils.neodb import App

app = App()

@dataclass
class TraversalState:
    node: Node
    parent_id: str
    level: int
    processed: bool = False

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class StructureDetails(TypedDict):
    classes: List[Dict]
    functions: List[Dict]
    api_endpoints: List[Dict]

class CodeAnalysisState(TypedDict):
    file_path: str
    project_root: str
    file_content: str
    parsed_content: Optional[Node] 
    imports: Dict[List, str]
    structure: StructureDetails

class PathResolver:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()

    def get_relative_path(self, absolute_path: str) -> str:
        return str(Path(absolute_path).resolve().relative_to(self.project_root))

    def resolve_import(self, import_path: str, current_file: str) -> str:
        current_dir = Path(current_file).parent
        return str((current_dir / import_path.replace('.', '/')).with_suffix('.py'))
    
relative_path = PathResolver(os.getenv('projectROOT'))

def get_function_name(node: Node) -> str:
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode()
    return ''

def get_function_parameters(node: Node) -> List[str]:
    params = []
    for child in node.children:
        if child.type == 'parameters':
            for param in child.children:
                if param.type == 'identifier':
                    params.append(param.text.decode())
    return params

def get_return_type(node: Node) -> str:
    for child in node.children:
        if child.type == 'type':
            return child.text.decode()
        if child.type == 'function_definition' and child.next_sibling:
            return child.next_sibling.text.decode()
    return ''

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
            

def read_and_parse(state: CodeAnalysisState) -> CodeAnalysisState:
    try:
        print(" -----XXXX----- ")
        with open(state['file_path'], 'r') as f:
            state['file_content'] = f.read()
            print(state['file_content'])

        
        # state['parsed_content'] = parser.parse(bytes(state['file_content'], 'utf8'))
        state['parsed_content'] = parser.parse(state['file_content'], 'utf8')

        if 'project_root' not in state:
            state['project_root'] = str(Path(state['file_path']).parent)

        path_resolver = PathResolver(state['project_root'])
        state['imports'] = get_imports(state['file_content'])
    except Exception as e:
        print(f"Parsing failed: {str(e)}")
        state['parsed_content'] = None
    return state

def analyze_state(state: CodeAnalysisState) -> CodeAnalysisState:
    return



def traverse_tree(root_node: TraversalState, imports: dict, file_path: str) -> List[Tuple[str, str, str]]:
    """
    Perform pre-order DFS traversal using a tree-sitter cursor.
    Returns a list of relationships (source_id, target_id, relationship_type)
    """
    
    stack = [TraversalState(root_node, parent_id="FILE:c/readStrcut:EXT:py", level=1, processed=False)]
    
    while stack:
        current_state = stack.pop()
        node = current_state.node
        parent_id = current_state.parent_id
        current_level = current_state.level
        # print(f"{node} -> {parent_id} -> {current_level} -> {current_node_id}  -> {current_state.processed}")
        # print('--------------------------XXXXX_-----------------')
        # If this state was already processed, print its start and end bytes and continue.
        if current_state.processed:
            # print(f"Processed node (id: {current_node_id}) - Start: {node.start_byte}, End: {node.end_byte}")
            continue

        if current_level == 1:
            current_level = current_level + 1

        if current_level >= 2:
            if node.type == "call":
                function_node = node.child_by_field_name("function")
                if function_node and function_node.type == "attribute":
                    method_name_node = function_node.child_by_field_name("attribute")
                    object_name_node = function_node.child_by_field_name("object")
                    if method_name_node and object_name_node:
                        # Decode node texts
                        method_name = method_name_node.text.decode("utf8")
                        object_name = object_name_node.text.decode("utf8")
                        # print("Detected call, object:", object_name, "method:", method_name)

                        if object_name == "requests" and method_name in {"get", "post", "put", "delete", "patch"}:
                            argument_node = node.child_by_field_name("arguments")
                            if argument_node and argument_node.child_count > 0:
                                route = None
                                for child in argument_node.children:
                                    if child.type in {"string", "string_literal"}:
                                        # Handle compound string nodes
                                        if child.child_count > 0:
                                            route = "".join(c.text.decode("utf8") for c in child.children)
                                        else:
                                            route = child.text.decode("utf8")
                                        break
                                if route is not None:
                                    route = route.strip("\"'")
                                    # print("Extracted route:", route)

                                target_id = f"APIENDPOINT:{route}:{method_name.upper()}"
                                            
                                if target_id and parent_id:
                                    # print(f"Detected API Call - method: {method_name.upper()}, route: {route}")
                                    # print(f"----------------------------> relation built {parent_id} --> {target_id}")
                                    pending_rels.add_relationship(parent_id, target_id, method_name.upper())

                        else:
                            target_id = None
                            for key in imports.keys():
                                if isinstance(key, tuple):
                                    if key[0] == object_name:
                                        target_id = f"FUNCTION:{imports[key]}:{object_name}"
                                        break
                                else:
                                    if key == object_name:
                                        target_id = f"FUNCTION:{imports[key]}:{object_name}"
                                        break

                            if target_id and parent_id:
                                # print(f"----------------------------> relation built {parent_id} --> {target_id}")
                                pending_rels.add_relationship(parent_id, target_id, 'CALLS')


                elif function_node.type == "identifier":
                        # Handle simple identifier calls (e.g., RD(), install_tree_sitter_language())
                        called_func = get_called_function_name(node)
            
                        target_id = None
                        for key in imports.keys():
                            if isinstance(key, tuple):
                                if key[0] == called_func:
                                    target_id = f"FUNCTION:{imports[key]}:{called_func}"
                                    break
                            else:
                                if key == called_func:
                                    target_id = f"FUNCTION:{imports[key]}:{called_func}"
                                    break

                        if target_id and parent_id:
                            # print(f"----------------------------> relation built {parent_id} --> {target_id}")
                            pending_rels.add_relationship(parent_id, target_id, 'CALLS')

            elif node.type == "decorated_definition":
                # Find the function inside the decorated definition
                function_node = None
                for child in node.children:
                    if child.type == "function_definition":
                        function_node = child
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
                func_name = get_function_name(node)
                params = get_function_parameters(node)
                return_type = get_return_type(node)
                relative_file_path = relative_path.get_relative_path(file_path) 
                node_id = f"FUNCTION:{relative_file_path}:{func_name}"
                # print(f"name: {func_name} ; params: {params} ; return_type : {return_type}")
                app.create_function_node(node_id, func_name, relative_file_path, params, return_type)
                # print(f"---------------------------->         node created {node_id}")
                fixed_parent_id = node_id
                current_state.processed = True
                if parent_id:
                    # print(f"---------------------------->         realtion built {node_id}  --> {current_state.parent_id}")
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






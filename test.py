#!/usr/bin/env python
import os
import sys
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Load the pre-compiled Python grammar.
PY_LANGUAGE = Language(tspython.language())

# Create and configure the parser.
parser = Parser(PY_LANGUAGE)
# parser.set_language(PY_LANGUAGE)

code = '''from tree_sitter import Language, Parser
import tree_sitter_python

# Load Python language
PY_LANGUAGE = Language(tree_sitter_python.language())

# Create a parser configured for Python
parser = Parser(PY_LANGUAGE)

def parse_python_code(source_code: str) -> tuple:
    """
    Parse Python source code and return the syntax tree with important nodes.
    
    Args:
        source_code: Python source code as a string
        
    Returns:
        tuple: (tree, root_node, function_defs, class_defs, calls)
    """
    # Convert source code to UTF-8 bytes
    code_bytes = source_code.encode('utf-8')
    
    # Parse the code
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    
    # Initialize collections
    function_defs = []
    class_defs = []
    calls = []
    
    # Recursive walk through nodes
    def walk(node):
        if node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            function_defs.append({
                'name': name_node.text.decode(),
                'start_line': name_node.start_point[0]+1,
                'end_line': node.end_point[0]+1
            })
        elif node.type == 'class_definition':
            name_node = node.child_by_field_name('name')
            class_defs.append({
                'name': name_node.text.decode(),
                'start_line': name_node.start_point[0]+1,
                'end_line': node.end_point[0]+1
            })
        elif node.type == 'call':
            function_node = node.child_by_field_name('function')
            calls.append({
                'name': function_node.text.decode(),
                'line': function_node.start_point[0]+1
            })
            
        for child in node.children:
            walk(child)
    
    walk(root_node)
    
    return tree, root_node, function_defs, class_defs, calls'''

def extract_import_nodes(code):
    """
    Parse the given source (as bytes) and return a list of nodes
    corresponding to import statements (both "import_statement" and
    "import_from_statement").
    """
    # code = code.encode("utf-8")
    tree = parser.parse(code)
    root = tree.root_node
    import_nodes = []

    def traverse(node):
        if node.type in ["import_statement", "import_from_statement"]:
            import_nodes.append(node)
        # Recursively traverse children.
        for child in node.children:
            traverse(child)

    traverse(root)
    return import_nodes

def process_file(file_path):
    """
    Process a single file, extracting and printing any import statements.
    """
    try:
        with open(file_path, "rb") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    import_nodes = extract_import_nodes(source)
    if import_nodes:
        # print(f"\nFile: {file_path}")
        for node in import_nodes:
            # Extract the source code for the node.
            import_statement = source[node.start_byte:node.end_byte].decode("utf8")
            print("  ", import_statement.strip())



process_file('test.py')

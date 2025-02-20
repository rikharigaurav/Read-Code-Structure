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

def traverse_tree(node, source_code, depth=0, parent_class=None):
    """Recursively traverse the syntax tree and extract structural information"""
    structural_info = []
    indent = '    ' * depth
    
    # Handle class definitions
    if node.type == 'class_definition':
        class_name = get_node_text(node.child_by_field_name('name'), source_code)
        structural_info.append(f"{indent}class {class_name} {{")
        for child in node.children:
            structural_info.extend(traverse_tree(child, source_code, depth+1, class_name))
        structural_info.append(f"{indent}}}")
    
    # Handle function definitions
    elif node.type == 'function_definition':
        func_name = get_node_text(node.child_by_field_name('name'), source_code)
        class_prefix = f"{parent_class}." if parent_class else ""
        structural_info.append(f"{indent}function {class_prefix}{func_name} {{")
        
        # Find function calls within the function body
        calls = find_function_calls(node, source_code)
        for call in calls:
            structural_info.append(f"{indent}    calls {call}")
        
        structural_info.append(f"{indent}}}")
    
    # Recursively process child nodes
    else:
        for child in node.children:
            structural_info.extend(traverse_tree(child, source_code, depth, parent_class))
    
    return structural_info

def get_node_text(node, source_code):
    """Get the text of a node from the source code"""
    return source_code[node.start_byte:node.end_byte].decode('utf8')

def find_function_calls(node, source_code):
    """Find function calls within a node"""
    calls = []
    if node.type == 'call':
        # Get the function name being called
        func_node = node.child_by_field_name('function')
        if func_node:
            calls.append(get_node_text(func_node, source_code))
    
    # Recursively check children for calls
    for child in node.children:
        calls.extend(find_function_calls(child, source_code))
    
    return calls

def generate_code_map(file_path):
    """Generate a structural map for a Python file"""
    with open(file_path, 'rb') as f:
        source_code = f.read()
    
    tree = parser.parse(source_code)
    root_node = tree.root_node
    
    structure = traverse_tree(root_node, source_code)
    return '\n'.join(structure)

if __name__ == "__main__":
    # Example usage
    file_path = 'test.py'
    code_map = generate_code_map(file_path)
    print(code_map)
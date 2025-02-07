import subprocess
import os
import sys
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Load the pre-compiled Python grammar.
PY_LANGUAGE = Language(tspython.language())

# Create and configure the parser.
parser = Parser(PY_LANGUAGE)

# Installs tree sitter languages
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


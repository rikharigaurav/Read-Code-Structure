import os
import importlib
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from utils.treeSitter import install_tree_sitter_language

tree_sitter_languages = {
    ".c": ["tree-sitter-c", "c"],
    ".cpp": ["tree-sitter-cpp", "cpp"],
    ".cs": ["tree-sitter-c-sharp", "csharp"],
    ".go": ["tree-sitter-go", "go"],
    ".hs": ["tree-sitter-haskell", "haskell"],
    ".java": ["tree-sitter-java", "java"],
    ".js": ["tree-sitter-javascript", "javascript"],
    ".jsx": ["tree-sitter-javascript", "javascript"],
    ".kt": ["tree-sitter-kotlin", "kotlin"],
    ".php": ["tree-sitter-php", "php"],
    ".py": ["tree-sitter-python", "python"],
    ".rb": ["tree-sitter-ruby", "ruby"],
    ".rs": ["tree-sitter-rust", "rust"],
    ".swift": ["tree-sitter-swift", "swift"],
    ".ts": ["tree-sitter-typescript", "typescript"],
    ".tsx": ["tree-sitter-typescript", "typescript"],
    ".lua": ["tree-sitter-lua", "lua"],
    ".r": ["tree-sitter-r", "r"],
    ".pl": ["tree-sitter-perl", "perl"],
    ".jl": ["tree-sitter-julia", "julia"],
    ".elm": ["tree-sitter-elm", "elm"],
    ".zig": ["tree-sitter-zig", "zig"],
    ".dart": ["tree-sitter-dart", "dart"],
    ".scm": ["tree-sitter-scheme", "scheme"],
    ".ml": ["tree-sitter-ocaml", "ocaml"],
    ".ex": ["tree-sitter-elixir", "elixir"],
    ".sh": ["tree-sitter-bash", "bash"],
    ".sql": ["tree-sitter-sql", "sql"],
    ".css": ["tree-sitter-css", "css"],
    ".html": ["tree-sitter-html", "html"],
    ".json": ["tree-sitter-json", "json"],
    ".yaml": ["tree-sitter-yaml", "yaml"],
    ".toml": ["tree-sitter-toml", "toml"],
    ".xml": ["tree-sitter-xml", "xml"],
    ".erl": ["tree-sitter-erlang", "erlang"],
    ".md": ["tree-sitter-markdown", "markdown"],
    ".nim": ["tree-sitter-nim", "nim"],
    ".txt": ["tree-sitter-plaintext", "plaintext"]
}



async def setFileParser(filePath: str, repoName: str):
    print("File path:", filePath)
    _, fileExt = os.path.splitext(filePath)
    print("File extension:", fileExt)

    if fileExt in tree_sitter_languages:
        filelanguage = tree_sitter_languages[fileExt]
        print({
            "tree-sitter-lang": filelanguage[0],
            "language": filelanguage[1],
        })
        try:
            tree_parser_import = install_tree_sitter_language(filelanguage[1])
            TSlang = importlib.import_module(tree_parser_import)
            print("The TS language", TSlang.__name__ )
            TApython = TSlang
            with open(filePath, 'r') as f:
                file_content = f.read()
            try:
                LANGAUAGE = Language(TApython.language())
                parser = Parser(LANGAUAGE)
                print(f"Parser loaded for {filelanguage[1]}")
                tree = parser.parse(bytes(file_content, "utf8"))
            except:
                if(fileExt == ".tsx"):
                    parser = Parser(TSlang.language_tsx())
                    print(f"Parser loaded for {fileExt}")
                elif fileExt == ".ts" :
                    parser = Parser(TSlang.language_typescript())
                    print(f"Parser loaded for {fileExt}")
                else :
                    print("Cannot find parser")

                tree = parser.parse(bytes(file_content, "utf8"))
        except FileNotFoundError:
            print(f"Error: Parser for {fileExt} not found or not compiled.")
        except RuntimeError as e:
            print(f"Unexpected error: {e}")
    else:
        print(f"Unsupported file extension: {fileExt}")



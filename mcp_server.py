"""
Read-Code-Structure — MCP Server
=================================
Exposes two tools so Claude can solve GitHub issues against the codebase
knowledge graph without any manual copy-paste:

  1. query_codebase_graph  — run Cypher against the Neo4j graph
  2. get_github_issue      — fetch a specific issue (+ comments)
  3. list_github_issues    — list open/closed issues for a repo

Run (stdio transport, for Claude Desktop / Claude Code):
    python mcp_server.py

Run (SSE/HTTP transport, for remote / web clients):
    python mcp_server.py --transport sse --port 8001
"""

import os
import sys
import json
import argparse
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ── Neo4j connection (lazy so the server boots even without credentials) ───────
_neo_db = None

def get_neo_db():
    global _neo_db
    if _neo_db is None:
        try:
            from utils.neodb import app as db
            _neo_db = db
        except Exception as e:
            raise RuntimeError(
                f"Neo4j connection failed: {e}\n"
                "Make sure NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD are set in .env"
            )
    return _neo_db


# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name="read-code-structure",
    instructions="""
You have access to an AI-powered codebase analysis system for the repository
that has been indexed into a Neo4j knowledge graph.

Workflow for solving a GitHub issue efficiently:
  1. Call list_github_issues to see what is open (if you don't know the number).
  2. Call get_github_issue with the specific issue number to get the full description
     and any comments — this is your problem statement.
  3. Call query_codebase_graph with targeted Cypher queries to locate the relevant
     files, functions, and API endpoints in the graph.
  4. Combine what you learned from the issue and the graph to produce a solution.

Tip: start narrow — search for a specific function or file name from the error
message before exploring broad paths.
""",
)


# ── Graph schema (embedded in tool description so Claude knows the vocabulary) ─
_SCHEMA = """
NODE LABELS & KEY PROPERTIES
─────────────────────────────
DataFile           file_name, file_path, file_ext, summary
DocumentationFile  file_name, file_path, file_ext, summary
Folder             folder_name, directory_path, summary
Function           function_name, file_path, return_type, summary
ApiEndpoint        endpoint_name, http_method, summary
TemplateMarkupFile file_name, file_path, file_ext, summary
TestingFile        file_name, file_path, test_framework, summary

RELATIONSHIPS
─────────────
(child)     -[:BELONGS_TO]->  (parent)      file→folder, fn→file, subfolder→folder
(fn)        -[:CALLS]->       (fn2)         function calls another function
(test_file) -[:TEST]->        (fn|endpoint) test covers a function or endpoint
(file)      -[:GET|POST|PUT|DELETE|PATCH]-> (ApiEndpoint)  HTTP calls in code

EXAMPLE QUERIES
───────────────
-- All functions in a file
MATCH (fn:Function)-[:BELONGS_TO]->(f:DataFile {file_name: 'auth.py'}) RETURN fn

-- What calls a given function
MATCH (caller:Function)-[:CALLS]->(fn:Function {function_name: 'validate_token'}) RETURN caller, fn

-- All API endpoints with their HTTP methods
MATCH (e:ApiEndpoint) RETURN e.endpoint_name, e.http_method ORDER BY e.http_method LIMIT 30

-- Files with the most outgoing function calls
MATCH (f:DataFile)<-[:BELONGS_TO]-(fn:Function)-[:CALLS]->()
RETURN f.file_name, count(*) AS calls ORDER BY calls DESC LIMIT 10

-- Test coverage for a module
MATCH p=(t:TestingFile)-[:TEST]->(n)-[:BELONGS_TO*0..1]->(f:DataFile)
WHERE f.file_name CONTAINS 'auth'
RETURN p LIMIT 20

-- Folder structure (top 2 levels)
MATCH p=(:Folder)-[:BELONGS_TO*1..2]->(:Folder) RETURN p LIMIT 30
"""


# ── Tool 1: Graph query ────────────────────────────────────────────────────────
@mcp.tool()
def query_codebase_graph(cypher: str) -> dict[str, Any]:
    """Query the codebase knowledge graph with a Cypher statement.

    Use this to explore code structure: find functions, trace call chains,
    locate API endpoints, check test coverage, and understand how files
    and folders are connected.

    Graph schema (node labels, properties, and relationships):

    NODE LABELS & KEY PROPERTIES
    DataFile           file_name, file_path, file_ext, summary
    DocumentationFile  file_name, file_path, file_ext, summary
    Folder             folder_name, directory_path, summary
    Function           function_name, file_path, return_type, summary
    ApiEndpoint        endpoint_name, http_method, summary
    TemplateMarkupFile file_name, file_path, file_ext, summary
    TestingFile        file_name, file_path, test_framework, summary

    RELATIONSHIPS
    (child)-[:BELONGS_TO]->(parent)      file to folder, fn to file
    (fn)-[:CALLS]->(fn2)                 function calls another function
    (test)-[:TEST]->(fn|endpoint)        test covers a function or endpoint
    (file)-[:GET|POST|PUT|DELETE|PATCH]->(ApiEndpoint)

    Args:
        cypher: A valid Cypher query string. Always include a LIMIT clause.
    """
    try:
        db = get_neo_db()
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    try:
        raw = db.run_query(cypher)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "hint": (
                "Check Cypher syntax and node labels. "
                "Valid labels: DataFile, DocumentationFile, Folder, "
                "Function, ApiEndpoint, TemplateMarkupFile, TestingFile."
            ),
        }

    if raw is None:
        return {"success": False, "error": "Query returned no result (possible syntax error)."}

    return {
        "success": True,
        "count": len(raw),
        "results": _serialize(raw),
    }


# ── Tool 2: Fetch one issue ────────────────────────────────────────────────────
@mcp.tool()
async def get_github_issue(repo: str, issue_number: int) -> dict[str, Any]:
    """Fetch a specific GitHub issue including its full body and comments.

    Use this after list_github_issues to get the complete problem description
    before querying the codebase graph.

    Args:
        repo: Repository in 'owner/repo' format  (e.g. 'octocat/Hello-World')
        issue_number: The issue number shown on GitHub (e.g. 42)
    """
    headers = _github_headers()
    base = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(base, headers=headers)
        if resp.status_code == 404:
            return {"success": False, "error": f"Issue #{issue_number} not found in {repo}."}
        if resp.status_code == 403:
            return {
                "success": False,
                "error": "GitHub rate limit hit or repo is private. Set GITHUB_TOKEN in .env.",
            }
        resp.raise_for_status()
        issue = resp.json()

        comments: list[dict] = []
        if issue.get("comments", 0) > 0:
            c_resp = await client.get(f"{base}/comments", headers=headers)
            if c_resp.status_code == 200:
                comments = [
                    {
                        "author": c["user"]["login"],
                        "created_at": c["created_at"],
                        "body": c["body"],
                    }
                    for c in c_resp.json()[:10]  # cap to avoid context overflow
                ]

    return {
        "success": True,
        "number": issue["number"],
        "title": issue["title"],
        "state": issue["state"],
        "author": issue["user"]["login"],
        "created_at": issue["created_at"],
        "labels": [lb["name"] for lb in issue.get("labels", [])],
        "assignees": [a["login"] for a in issue.get("assignees", [])],
        "body": issue.get("body") or "(no description provided)",
        "comments_count": issue.get("comments", 0),
        "comments": comments,
        "url": issue["html_url"],
    }


# ── Tool 3: List issues ────────────────────────────────────────────────────────
@mcp.tool()
async def list_github_issues(
    repo: str,
    state: str = "open",
    limit: int = 20,
) -> dict[str, Any]:
    """List GitHub issues for a repository.

    Use this to get an overview before picking a specific issue to investigate
    with get_github_issue.

    Args:
        repo:  Repository in 'owner/repo' format  (e.g. 'octocat/Hello-World')
        state: 'open', 'closed', or 'all'  (default 'open')
        limit: Number of issues to return, 1-50  (default 20)
    """
    limit = max(1, min(limit, 50))
    headers = _github_headers()
    url = (
        f"https://api.github.com/repos/{repo}/issues"
        f"?state={state}&per_page={limit}&sort=updated&direction=desc"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 404:
            return {"success": False, "error": f"Repository '{repo}' not found."}
        if resp.status_code == 403:
            return {
                "success": False,
                "error": "GitHub rate limit hit or repo is private. Set GITHUB_TOKEN in .env.",
            }
        resp.raise_for_status()
        issues = resp.json()

    return {
        "success": True,
        "repo": repo,
        "state": state,
        "count": len(issues),
        "issues": [
            {
                "number": i["number"],
                "title": i["title"],
                "state": i["state"],
                "author": i["user"]["login"],
                "labels": [lb["name"] for lb in i.get("labels", [])],
                "comments": i.get("comments", 0),
                "updated_at": i["updated_at"],
                "url": i["html_url"],
            }
            for i in issues
            # GitHub returns PRs in the issues endpoint — filter them out
            if "pull_request" not in i
        ],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _serialize(obj: Any) -> Any:
    """
    Recursively make Neo4j / arbitrary Python values JSON-safe.
    neo4j.graph.Integer is a subclass of int but json.dumps rejects it in
    some driver versions — calling int() on it always works.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)          # normalises neo4j.graph.Integer
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    # Fallback for any remaining Neo4j types
    try:
        return int(obj)
    except (TypeError, ValueError):
        return str(obj)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-Code-Structure MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport to use (default: stdio for Claude Desktop / Claude Code)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for SSE transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port for SSE transport (default: 8001)"
    )
    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting MCP server (SSE) on http://{args.host}:{args.port}", flush=True)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # stdio — Claude spawns this process and talks over stdin/stdout
        mcp.run(transport="stdio")

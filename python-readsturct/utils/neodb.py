from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable
import json
import os
import hashlib

class CodebaseGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Close the driver connection when done
        self.driver.close()

    def clear_database(self):
        with self.driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All nodes and edges are removed")

    def set_constraints(self):
        # Ensure each node has unique identifiers
        with self.driver.session(database="neo4j") as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (F:Folder) REQUIRE F.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Class) REQUIRE cl.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (api:API) REQUIRE api.id IS UNIQUE")
            print("Constraints set for unique node identifiers")

    def create_file(self, file_path, file_type):
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_file_node, 
                file_path, 
                file_type
            )

    def create_function(self, function_name, file_path, access_modifier, lines_of_code):
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_function_node, 
                function_name, 
                file_path, 
                access_modifier, 
                lines_of_code
            )

    def create_class(self, class_name, file_path):
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_class_node, 
                class_name, 
                file_path
            )

    def create_api(self, method, endpoint):
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_api_node, 
                method, 
                endpoint
            )

    def create_relationship(self, from_node_id, to_node_id, relationship_type):
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_relationship, 
                from_node_id, 
                to_node_id, 
                relationship_type
            )

    def count_nodes_edges(self):
        with self.driver.session(database="neo4j") as session:
            nodes = session.run("MATCH (n) RETURN count(n) as nodes").single()["nodes"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) as edges").single()["edges"]
            print(f"Number of nodes: {nodes}")
            print(f"Number of edges: {edges}")

    # Static methods for transaction functions
    @staticmethod
    def _create_file_node(tx, file_path, file_type):
        file_id = hashlib.sha256(file_path.encode()).hexdigest()
        query = (
            "MERGE (f:File { id: $file_id }) "
            "SET f.path = $file_path, f.type = $file_type "
            "RETURN f"
        )
        tx.run(query, file_id=file_id, file_path=file_path, file_type=file_type)

    @staticmethod
    def _create_function_node(tx, function_name, file_path, access_modifier, lines_of_code):
        function_id = hashlib.sha256(f"{function_name}:{file_path}".encode()).hexdigest()
        query = (
            "MERGE (fn:Function { id: $function_id }) "
            "SET fn.name = $function_name, fn.file_path = $file_path, "
            "fn.access_modifier = $access_modifier, fn.lines_of_code = $lines_of_code "
            "RETURN fn"
        )
        tx.run(query, function_id=function_id, function_name=function_name, file_path=file_path,
               access_modifier=access_modifier, lines_of_code=lines_of_code)

    @staticmethod
    def _create_class_node(tx, class_name, file_path):
        class_id = hashlib.sha256(f"{class_name}:{file_path}".encode()).hexdigest()
        query = (
            "MERGE (cl:Class { id: $class_id }) "
            "SET cl.name = $class_name, cl.file_path = $file_path "
            "RETURN cl"
        )
        tx.run(query, class_id=class_id, class_name=class_name, file_path=file_path)

    @staticmethod
    def _create_api_node(tx, method, endpoint):
        api_id = hashlib.sha256(f"{method}:{endpoint}".encode()).hexdigest()
        query = (
            "MERGE (api:API { id: $api_id }) "
            "SET api.method = $method, api.endpoint = $endpoint "
            "RETURN api"
        )
        tx.run(query, api_id=api_id, method=method, endpoint=endpoint)

    @staticmethod
    def _create_relationship(tx, from_node_id, to_node_id, relationship_type):
        query = (
            "MATCH (a { id: $from_node_id }), (b { id: $to_node_id }) "
            "MERGE (a)-[r:{relationship_type}]->(b) "
            "RETURN type(r)"
        )
        tx.run(query, from_node_id=from_node_id, to_node_id=to_node_id, relationship_type=relationship_type.upper())


if __name__ == "__main__":
    # Load environment variables
    uri = os.getenv('NEO4J_URI')
    user = "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    
    # Initialize the app
    app = CodebaseGraph(uri, user, password)
    
    # Clear existing nodes and relationships
    app.clear_database()
    app.set_constraints()
    
    # Load sample data
    with open('sample.json') as f:
        data = json.load(f)
    
    # Create nodes and relationships based on the data
    for file_data in data['files']:
        app.create_file(file_data['path'], file_data['type'])
    
    for function_data in data['functions']:
        app.create_function(function_data['name'], function_data['file_path'], function_data['access_modifier'], function_data['lines_of_code'])
    
    for class_data in data['classes']:
        app.create_class(class_data['name'], class_data['file_path'])

    for api_data in data['apis']:
        app.create_api(api_data['method'], api_data['endpoint'])
    
    # Create relationships based on the data
    for edge_data in data['edges']:
        app.create_relationship(edge_data['source'], edge_data['target'], edge_data['relationship'])
    
    # Display the final counts
    app.count_nodes_edges()
    app.close()

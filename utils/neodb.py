from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable
import json
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()
class App:
    def __init__(self, uri=None, user=None, password=None):
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        URI = os.getenv("NEO4J_URI")
        AUTH = (user, password)

        with GraphDatabase.driver(URI, auth=AUTH) as self.driver:
            self.driver.verify_connectivity()
            print("connected succesfully")

    def generate_node_hash(file_path: str, file_name: str) -> str:
        """
        Generate a unique hash ID for a node based on the file path and file name.

        Parameters:
            file_path (str): The path of the file.
            file_name (str): The name of the file.

        Returns:
            str: A unique hash ID for the node.
        """
        # Combine file path and file name into a single string
        unique_string = f"{file_path}/{file_name}"
        
        # Generate SHA-256 hash
        hash_object = hashlib.sha256(unique_string.encode('utf-8'))
        hash_id = hash_object.hexdigest()
        
        return hash_id

    def create_data_file_node(self, file_id, file_name, file_path, file_ext):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(
                self._create_data_file_node, file_id, file_name, file_path, file_ext
            )
            return result

    @staticmethod
    def _create_data_file_node(tx, file_id, file_name, file_path, file_ext):
        query = (
            "MERGE (df:DataFile { id: $file_id, file_name: $file_name, file_path: $file_path, "
            "file_ext: $file_ext}) "
            "SET df.vector_id = NULL, df.summary = '' "  # Added empty summary
            "RETURN df"
        )
        result = tx.run(
            query, file_id=file_id, file_name=file_name, file_path=file_path, file_ext=file_ext
        )
        try:
            record = result.single()
            if record:
                return {"node_id": record["df"]["id"], "node_type": "DataFile"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    # def create_graph_file_node(self, file_id, file_name, file_path, file_ext, file_type):
    #     with self.driver.session(database="neo4j") as session:
    #         result = session.write_transaction(
    #             self._create_graph_file_node, file_id, file_name, file_path, file_ext, file_type
    #         )
    #         return result

    # @staticmethod
    # def _create_graph_file_node(tx, file_id, file_name, file_path, file_ext, file_type):
    #     query = (
    #         "MERGE (gf:GraphFile { id: $file_id, file_name: $file_name, file_path: $file_path, "
    #         "file_ext: $file_ext, file_type: $file_type }) "
    #         "SET gf.vector_id = NULL, gf.summary = '' "  # Added empty summary
    #         "RETURN gf"
    #     )
    #     result = tx.run(
    #         query, file_id=file_id, file_name=file_name, file_path=file_path, file_ext=file_ext, file_type=file_type
    #     )
    #     try:
    #         record = result.single()
    #         if record:
    #             return {"node_id": record["gf"]["id"], "node_type": "GraphFile"}
    #         else:
    #             return None
    #     except Exception as e:
    #         print(f"Query failed: {e}")
    #         return None

    def create_template_markup_file_node(self, file_id, file_name, file_path, file_ext, dependencies):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(
                self._create_template_markup_file_node, file_id, file_name, file_path, file_ext, dependencies
            )
            return result

    @staticmethod
    def _create_template_markup_file_node(tx, file_id, file_name, file_path, file_ext, dependencies):
        query = (
            "MERGE (tmf:TemplateMarkupFile { id: $file_id, file_name: $file_name, file_path: $file_path, "
            "file_ext: $file_ext, dependencies: $dependencies }) "
            "SET tmf.vector_id = NULL, tmf.summary = '' "  # Added empty summary
            "RETURN tmf"
        )
        result = tx.run(
            query, file_id=file_id, file_name=file_name, file_path=file_path, file_ext=file_ext,
            dependencies=dependencies
        )
        try:
            record = result.single()
            if record:
                return {"node_id": record["tmf"]["id"], "node_type": "TemplateMarkupFile"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def create_testing_file_node(self, file_id, file_name, file_path, file_ext, test_framework, test_reference_dict):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(
                self._create_testing_file_node, file_id, file_name, file_path, file_ext, test_framework, test_reference_dict
            )
            return result

    @staticmethod
    def _create_testing_file_node(tx, file_id, file_name, file_path, file_ext, test_framework, test_reference_dict):
        query = (
            "MERGE (tf:TestingFile { id: $file_id, file_name: $file_name, file_path: $file_path, "
            "file_ext: $file_ext, test_framework: $test_framework, test_reference_dict: $test_reference_dict }) "
            "SET tf.vector_id = NULL, tf.summary = '' "  # Added empty summary
            "RETURN tf"
        )
        result = tx.run(
            query, file_id=file_id, file_name=file_name, file_path=file_path, file_ext=file_ext,
            test_framework=test_framework, test_reference_dict=test_reference_dict
        )
        try:
            record = result.single()
            if record:
                return {"node_id": record["tf"]["id"], "node_type": "TestingFile"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def create_documentation_file_node(self, file_id, file_name, file_path, file_ext, purpose):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(
                self._create_documentation_file_node, file_id, file_name, file_path, file_ext,  purpose
            )
            return result

    @staticmethod
    def _create_documentation_file_node(tx, file_id, file_name, file_path, file_ext, purpose):
        query = (
            "MERGE (df:DocumentationFile { id: $file_id, file_name: $file_name, file_path: $file_path, "
            "file_ext: $file_ext, purpose: $purpose }) "
            "SET df.vector_id = NULL, df.summary = '' "  # Added empty summary
            "RETURN df"
        )
        result = tx.run(
            query, file_id=file_id, file_name=file_name, file_path=file_path, file_ext=file_ext,
            purpose=purpose
        )
        try:
            record = result.single()
            if record:
                return {"node_id": record["df"]["id"], "node_type": "DocumentationFile"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None
        

    def create_api_endpoint_node(self, nodeID, url, http_method):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(self._create_api_endpoint_node, nodeID, url, http_method)
            return result

    @staticmethod
    def _create_api_endpoint_node(tx, nodeID, url, http_method):
        query = (
            "MERGE (ae:APIEndpoint { id: $nodeID, endpoint: $url, http_method: $http_method }) "
            "SET ae.vector_format = NULL "
            "RETURN ae"
        )
        result = tx.run(query, nodeID=nodeID, url=url, http_method=http_method)
        try:
            record = result.single()
            if record:
                return {"node_id": record["ae"]["id"], "node_type": "APIEndpoint"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def create_function_node(self, function_id, function_name, file_path, return_type=""):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(
                self._create_function_node, 
                function_id, 
                function_name, 
                file_path, 
                return_type
            )
            return result

    @staticmethod
    def _create_function_node(tx, function_id, function_name, file_path, return_type):
        
        query = (
            "MERGE (f:Function { id: $function_id }) "
            "SET f.function_name = $function_name, "
            "f.file_path = $file_path, "
            "f.vector_format = null, "
            "f.return_type = $return_type "
            "RETURN f"
        )
        result = tx.run(
            query,
            function_id=function_id,
            function_name=function_name,
            file_path=file_path,
            return_type=str(return_type)  # Ensure string type
        )
        try:
            record = result.single()
            if record:
                return {"node_id": record["f"]["id"], "node_type": "Function"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None


    def create_folder_node(self, folder_id, folder_name, directory_path):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(self._create_folder_node, folder_id, folder_name, directory_path)
            return result

    @staticmethod
    def _create_folder_node(tx, folder_id, folder_name, directory_path):
        query = (
            "MERGE (f:Folder { id: $folder_id, folder_name: $folder_name, directory_path: $directory_path }) "
            "SET f.vector_id = NULL, f.summary = '' "  # Added empty summary
            "RETURN f"
        )
        result = tx.run(query, folder_id=folder_id, folder_name=folder_name, directory_path=directory_path)
        try:
            record = result.single()
            if record:
                return {"node_id": record["f"]["id"], "node_type": "Folder"}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def create_relation(self, child_id, parent_id, relation="BELONGS_TO"):
        """
        Creates a directional relationship from the child node to the parent node in Neo4j.
        
        Parameters:
        - child_id (str): The ID of the child node.
        - parent_id (str): The ID of the parent node.
        - relation (str): The type of relationship to create (default is "BELONGS_TO").
        """
        with self.driver.session(database="neo4j") as session:
            session.write_transaction(
                self._create_and_return_relation, child_id, parent_id, relation
            )
        
    @staticmethod
    def _create_and_return_relation(tx, child_id, parent_id, relation):
        """
        Helper function to execute the Cypher query for creating the relationship.
        """
        query = (
            f"MATCH (child {{id: $child_id}}), (parent {{id: $parent_id}}) "
            f"MERGE (child)-[r:{relation}]->(parent) "
            f"RETURN child.id AS child, parent.id AS parent"
        )
        
        tx.run(query, child_id=child_id, parent_id=parent_id)


            
    def get_incoming_nodes(self, node_id):
        with self.driver.session(database="neo4j") as session:
            result = session.read_transaction(self._fetch_incoming_nodes, node_id)
            return result

    @staticmethod
    def _fetch_incoming_nodes(tx, node_id):
        query = (
            "MATCH (source)-[r]->(target {id: $node_id}) "
            "RETURN source.id AS source_id, labels(source) AS labels"
        )
        result = tx.run(query, node_id=node_id)
        try:
            return [{"source_id": row["source_id"], "labels": row["labels"]} for row in result]
        except Exception as e:
            print(f"Query failed: {e}")
            return []
        
    def get_node_context(self, node_id):
        with self.driver.session(database="neo4j") as session:
            result = session.read_transaction(self._fetch_node_context, node_id)
            return result

    @staticmethod
    def _fetch_node_context(tx, node_id):
        query = (
            "MATCH (n {id: $node_id}) "
            "RETURN n.context AS context"
        )
        result = tx.run(query, node_id=node_id)
        # try:
        #     record = result.single()
        #     return record["context"] if record else None
        # except Exception as e:
        #     print(f"Query failed: {e}")
        #     return None

    def get_node_by_id(self, node_id):
        with self.driver.session() as session:
            result = session.read_transaction(self._fetch_node_by_id, node_id)
            return bool(result)  

    @staticmethod
    def _fetch_node_by_id(tx, node_id):
        query = (
            "MATCH (n) "
            "WHERE n.id = $node_id "
            "RETURN n IS NOT NULL AS node_exists"
        )
        try:
            result = tx.run(query, node_id=node_id)
            record = result.single()
            return record["node_exists"] if record else False
        except Exception as e:
            print(f"Query failed: {e}")
            return False

    def update_node_vector_format(self, node_id, vector_format, node_label):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(self._update_vector_format, node_id, vector_format, node_label)
            return result
    @staticmethod
    def _update_vector_format(tx, node_id, vector_format, node_label):
        # Safely interpolate the node_label into the query
        query = (
            f"MATCH (n:{node_label} {{id: $node_id}}) "  # Dynamic label
            "SET n.vector_format = $vector_format "
            "RETURN n.id AS node_id, n.vector_format AS updated_vector_format"
        )
        result = tx.run(query, node_id=node_id, vector_format=vector_format)
        try:
            record = result.single()
            if record:
                return {"node_id": record["node_id"], "updated_vector_format": record["updated_vector_format"]}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def update_summary_context(self, node_id, summary_context):
        with self.driver.session(database="neo4j") as session:
            result = session.write_transaction(self._update_summary_context, node_id, summary_context)
            return result

    @staticmethod
    def _update_summary_context(tx, node_id, summary_context):
        query = (
            "MATCH (n {id: $node_id}) "
            "SET n.summary_context = $summary_context "
            "RETURN n.id AS node_id, n.summary_context AS updated_summary_context"
        )
        result = tx.run(query, node_id=node_id, summary_context=summary_context)
        try:
            record = result.single()
            if record:
                return {"node_id": record["node_id"], "updated_summary_context": record["updated_summary_context"]}
            else:
                return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None


    def remove_all(self):
        with self.driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All nodes and edges are removed")

        # set the constraint for id
        with self.driver.session(database="neo4j") as session:
            session.run("CREATE CONSTRAINT unique_profile_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
            print("Constraint is set")


app = App()

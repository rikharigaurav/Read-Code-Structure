from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from hashlib import md5
from pinecone import Pinecone, ServerlessSpec
import os
import re
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import time
import json

load_dotenv()

class pineconeOperation:
    def __init__(self, index):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        self.cohere_keys = [
            os.getenv("COHERE_API_KEY_1"),
            os.getenv("COHERE_API_KEY_2"),
            os.getenv("COHERE_API_KEY_3"),
            os.getenv("COHERE_API_KEY_4"),
            os.getenv("COHERE_API_KEY_5"),
            os.getenv("COHERE_API_KEY_6"),
            os.getenv("COHERE_API_KEY_7"),
            os.getenv("COHERE_API_KEY_8")
        ]
        self.current_cohere_key_index = 0
        
        self.mistral_keys = [
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("MISTRAL_API_KEY")
        ]
        self.current_mistral_key_index = 0
        
        self.current_embedding_type = "cohere"
        self.embeddings = self._get_cohere_embeddings()
        self.index_name = index
        
        self.code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.html', '.css', '.ts', '.sh', '.json', '.xml', '.yaml', '.yml', '.sql', '.rs', '.swift', '.kt']

        self._print_api_key_info()

    def _print_api_key_info(self):
        """Print information about available API keys"""
        
        valid_cohere_keys = sum(1 for key in self.cohere_keys if key and len(key) > 10)
        print(f"Found {valid_cohere_keys} potentially valid Cohere API keys")
        
        unique_mistral_keys = len(set([key for key in self.mistral_keys if key and len(key) > 10]))
        print(f"Found {unique_mistral_keys} unique potentially valid Mistral API keys")

    def _get_cohere_embeddings(self):
        """Create a CohereEmbeddings instance with the current key"""
        return CohereEmbeddings(
            cohere_api_key=self.cohere_keys[self.current_cohere_key_index],
            model="embed-english-v3.0"
        )
    
    def _get_mistral_embeddings(self):
        """Create a MistralAIEmbeddings instance with the current key"""
        return MistralAIEmbeddings(
            api_key=self.mistral_keys[self.current_mistral_key_index],
            model="mistral-embed"  
        )

    def _is_code_file(self, file_path):
        """Check if the file is a source code file based on extension"""
        if not file_path:
            return False
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.code_extensions

    def _select_embeddings_model(self, file_path=None):
        """Select appropriate embedding model based on file type - ALWAYS USE COHERE"""
        
        if self.current_embedding_type != "cohere":
            self.current_embedding_type = "cohere"
            self.embeddings = self._get_cohere_embeddings()
            print("Using Cohere model for all files")
        
        return self.embeddings

    def _rotate_cohere_key(self):
        """Rotate to the next Cohere API key"""
        self.current_cohere_key_index = (self.current_cohere_key_index + 1) % len(self.cohere_keys)
        print(f"Rotating to Cohere API key #{self.current_cohere_key_index + 1}")
        if self.current_embedding_type == "cohere":
            self.embeddings = self._get_cohere_embeddings()
    
    def _rotate_mistral_key(self):
        """Rotate to the next Mistral API key"""
        self.current_mistral_key_index = (self.current_mistral_key_index + 1) % len(self.mistral_keys)
        print(f"Rotating to Mistral API key #{self.current_mistral_key_index + 1}")
        if self.current_embedding_type == "mistral":
            self.embeddings = self._get_mistral_embeddings()

    def convert_to_serializable(self, data):
        """Convert non-serializable tuple keys to strings."""
        if isinstance(data, dict):
            return {str(k): self.convert_to_serializable(v) for k, v in data.items()}
        return data

    def get_embeddings(self, text: str, file_path=None):
        """Get embeddings with automatic key rotation on failure"""
        self._select_embeddings_model(file_path)
        
        keys_tried = 0
        while keys_tried < len(self.cohere_keys):
            try:
                
                if len(text) > 8000:
                    print(f"Text too long ({len(text)} chars), truncating to 8000 chars")
                    text = text[:8000]
                
                res = self.embeddings.embed_documents([text])
                return res
            except Exception as error:
                error_str = str(error).lower()
                print(f"Cohere embedding error: {error_str}")
                
                if "rate" in error_str or "limit" in error_str or "quota" in error_str or error_str == "'data'" or "429" in error_str:
                    print(f"Cohere API key #{self.current_cohere_key_index + 1} exhausted. Trying next key...")
                    self._rotate_cohere_key()
                    keys_tried += 1
                    
                    time.sleep(2)
                else:
                    print("Error calling Cohere embeddings API:", error)
                    raise error
        
        raise Exception("All Cohere API keys have been exhausted")

    def embed_text(self, text, file_path=None, metadata=None):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try: 
                
                if len(text) < 10:
                    print(f"Warning: Very short text ({len(text)} chars)")
                    
                    if len(text) < 3:
                        text = text + " " * (10 - len(text))
                
                embeddings = self.get_embeddings(text, file_path)
                hash_id = md5(text.encode()).hexdigest()

                # print(f"{len(embeddings)} embeddings generated for text of length {len(text)} chars")

                # Create base record with only allowed Pinecone keys
                record = {
                    "id": hash_id,
                    "values": embeddings[0],
                    "metadata": {
                        "text": self.truncate_string_by_bytes(text, 3600)
                    }
                }
                
                # Add all metadata fields under the metadata key
                if metadata is not None:
                    for key, value in metadata.items():
                        # Convert key to string
                        str_key = str(key)
                        
                        # Handle different value types
                        if isinstance(value, (dict, list, tuple)):
                            # Convert complex types to JSON string
                            record["metadata"][str_key] = json.dumps(value)
                        elif isinstance(value, (str, int, float, bool)):
                            # Keep primitive types as-is
                            record["metadata"][str_key] = value
                        elif value is None:
                            # Keep None values
                            record["metadata"][str_key] = None
                        else:
                            # Convert other types to string
                            record["metadata"][str_key] = str(value)

                # print(f"Metadata added to record: {record['metadata']}")

                return record
                
            except Exception as error:
                error_str = str(error).lower()
                if "all mistral api keys have been exhausted" in error_str or "all cohere api keys have been exhausted" in error_str:
                    
                    print("Error embedding text: All API keys exhausted")
                    raise error
                
                retry_count += 1
                print(f"Error embedding text (attempt {retry_count}/{max_retries}): {error}")
                if retry_count >= max_retries:
                    raise error
                
                time.sleep(2)

    def truncate_string_by_bytes(self, string, max_bytes):
        encoded = string.encode("utf-8")
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def prepare_document(self, content, chunk_size=800, chunk_overlap=50):
        try:
            
            content = re.sub(r'\n{3,}', '\n\n', content)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_text(content)
            print(f"Split document into {len(docs)} chunks")
            return docs
        except Exception as error:
            print("Error preparing document:", error)
            raise error

    def load_text_to_pinecone(self, file_id, file_path=None, metadata=None, content=None):
        print(f"Loading text to Pinecone with file_id: {file_id}, file_path: {file_path}, metadata: {metadata}")
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )
        namespace_name = file_id
        if file_path and not content:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Successfully read file: {file_path} ({len(content)} chars)")
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    print(f"Read file with latin-1 encoding: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    raise e
        else:
            namespace_name += ":ENG"
        
        try:
            texts = self.prepare_document(content)
            vectors = []
            
            successful_chunks = 0
            failed_chunks = 0
            
            for i, text in enumerate(texts):
                try:
                    print(f"Processing chunk {i+1}/{len(texts)} ({len(text)} chars)")
                    vector = self.embed_text(text, file_path, metadata)
                    vectors.append(vector)
                    successful_chunks += 1
                    
                    
                    if i < len(texts) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    failed_chunks += 1
                    if "All Cohere API keys have been exhausted" in str(e) or "All Mistral API keys have been exhausted" in str(e):
                        print(f"Failed to embed chunk: {e}")
                        
                        break
                    print(f"Error embedding chunk {i+1}: {e}")
                
            
            print(f"Processing complete: {successful_chunks} successful, {failed_chunks} failed chunks")
            
            if not vectors:
                raise Exception("No vectors were successfully created")

            
            print(f"Uploading {len(vectors)} vectors to Pinecone namespace: {namespace_name}")
            index = self.pc.Index(self.index_name)
            response = index.upsert(vectors=vectors, namespace=namespace_name)

            print(f"Successfully uploaded vectors to Pinecone: {response}")
            return namespace_name
        
        except Exception as error:
            print("Error in load_text_to_pinecone:", error)
            raise error

    def retrieve_data_from_pinecone(self, context, score_threshold=0.5, metadata_filter=None, namespace=None):
        index = self.pc.Index(self.index_name)
        print(f"Context to retrieve: {context[:50]}..." if len(context) > 50 else context)
        
        
        if self.current_embedding_type != "cohere":
            self.current_embedding_type = "cohere"
            self.embeddings = self._get_cohere_embeddings()
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                vector_store = PineconeVectorStore(
                    index=index,  
                    embedding=self.embeddings
                )
                retriever_kwargs = {
                    "search_type": "similarity_score_threshold",
                    "search_kwargs": {"k": 3,
                                       "score_threshold": score_threshold,
                                       }
                }
                
                if namespace:
                    retriever_kwargs["search_kwargs"]["namespace"] = namespace
                
                if metadata_filter:
                    retriever_kwargs["search_kwargs"]["filter"] = metadata_filter
                
                retriever = vector_store.as_retriever(**retriever_kwargs)
                
                result = retriever.invoke(context)

                # print(f"Retrieved result : {result} ")
                
                formatted_results = []
                for doc in result:
                    
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    
                    formatted_doc = {
                        'content': content,
                        'metadata': metadata
                    }
                    formatted_results.append(formatted_doc)

                # print(f"the retrieved data {formatted_results}")

                return formatted_results
                
            except Exception as error:
                error_str = str(error).lower()
                print(f"Retrieval error: {error_str}")
                
                if "rate" in error_str or "limit" in error_str or "quota" in error_str or error_str == "'data'" or "429" in error_str:
                    if self.current_cohere_key_index + 1 < len(self.cohere_keys):
                        print(f"Cohere API key #{self.current_cohere_key_index + 1} exhausted during retrieval. Trying next key...")
                        self._rotate_cohere_key()
                    else:
                        print(f"All Cohere keys exhausted during retrieval.")
                        retry_count = max_retries  
                
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed retrieval after {max_retries} attempts: {error}")
                    raise Exception(f"Failed to retrieve data after {max_retries} attempts: {error}")
                
                time.sleep(2)
        
        raise Exception("Unexpected error in retrieval function")

    def get_namespace_names(self):
        if not self.pc.has_index(self.index_name):
            print(f"Index {self.index_name} does not exist")
            return []
        
        index = self.pc.Index(host="https://repository-covk0y4.svc.aped-4627-b74a.pinecone.io")
        namespaces = index.describe_index_stats()
        # print(namespaces['namespaces'])
        return namespaces['namespaces']

    def delete_index(self):
        if self.pc.has_index(self.index_name):
            self.pc.delete_index(self.index_name)
            print("Index deleted successfully")

    def create_index(self):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )
            print("Index created successfully")
        else:
            print("Index already exists")

pinecone = pineconeOperation('repository')

if __name__ == "__main__":
    # Example usage
    try:
        namespace = pinecone.get_namespace_names()
        print(f"Retrieved namespaces: {namespace}")

    except Exception as e:
        print(f"Error: {e}")
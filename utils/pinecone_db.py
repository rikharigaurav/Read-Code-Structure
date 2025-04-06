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
        
        # Cohere keys and settings
        self.cohere_keys = [
            os.getenv("COHERE_API_KEY_1"),
            os.getenv("COHERE_API_KEY_2"),
            os.getenv("COHERE_API_KEY_3")
        ]
        self.current_cohere_key_index = 0
        
        # Mistral keys and settings
        self.mistral_keys = [
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("MISTRAL_API_KEY")
        ]
        self.current_mistral_key_index = 0
        
        # Default to Cohere embeddings always
        self.current_embedding_type = "cohere"
        self.embeddings = self._get_cohere_embeddings()
        self.index_name = index
        
        # Code file extensions to use Cohere for (previously used Mistral)
        self.code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.html', '.css', '.ts', '.sh', '.json', '.xml', '.yaml', '.yml', '.sql', '.rs', '.swift', '.kt']

        # Print API key info
        self._print_api_key_info()

    def _print_api_key_info(self):
        """Print information about available API keys"""
        # Check Cohere keys
        valid_cohere_keys = sum(1 for key in self.cohere_keys if key and len(key) > 10)
        print(f"Found {valid_cohere_keys} potentially valid Cohere API keys")
        
        # Check Mistral keys
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
            model="mistral-embed"  # Use standard embedding model instead of codestral
        )
    
    def _is_code_file(self, file_path):
        """Check if the file is a source code file based on extension"""
        if not file_path:
            return False
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.code_extensions
    
    def _select_embeddings_model(self, file_path=None):
        """Select appropriate embedding model based on file type - ALWAYS USE COHERE"""
        # Changed to always use Cohere since Mistral is having issues
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
        # Select appropriate model based on file type - always use Cohere now
        self._select_embeddings_model(file_path)
        
        # ONLY USE COHERE SINCE MISTRAL IS FAILING
        keys_tried = 0
        while keys_tried < len(self.cohere_keys):
            try:
                # Limit text length to prevent issues
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
                    # Add delay between requests
                    time.sleep(2)
                else:
                    print("Error calling Cohere embeddings API:", error)
                    raise error
        
        # If all Cohere keys are exhausted
        raise Exception("All Cohere API keys have been exhausted")

    def embed_text(self, text, file_path=None, file_structure=None):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try: 
                # Sanitize and prepare text
                if len(text) < 10:
                    print(f"Warning: Very short text ({len(text)} chars)")
                    # Pad very short text to avoid API issues
                    if len(text) < 3:
                        text = text + " " * (10 - len(text))
                
                embeddings = self.get_embeddings(text, file_path)

                hash_id = md5(text.encode()).hexdigest()
                metadata = {
                    "text": self.truncate_string_by_bytes(text, 3600)
                }
                
                if file_structure is not None:
                    metadata["File_Structure"] = file_structure
                    
                if file_path is not None:
                    metadata["file_path"] = file_path
                    metadata["is_code"] = self._is_code_file(file_path)

                record = {
                    "id": hash_id,
                    "values": embeddings[0],    
                    "metadata": metadata,
                }

                return record
            except Exception as error:
                error_str = str(error).lower()
                if "all mistral api keys have been exhausted" in error_str or "all cohere api keys have been exhausted" in error_str:
                    # If all keys are exhausted, propagate the error
                    print("Error embedding text: All API keys exhausted")
                    raise error
                
                retry_count += 1
                print(f"Error embedding text (attempt {retry_count}/{max_retries}): {error}")
                if retry_count >= max_retries:
                    raise error
                
                # Longer delay before retry
                time.sleep(2)

    def truncate_string_by_bytes(self, string, max_bytes):
        encoded = string.encode("utf-8")
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def prepare_document(self, content, chunk_size=800, chunk_overlap=50):
        try:
            # Preserve some newlines for better context
            content = re.sub(r'\n{3,}', '\n\n', content)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_text(content)
            print(f"Split document into {len(docs)} chunks")
            return docs
        except Exception as error:
            print("Error preparing document:", error)
            raise error

    def load_text_to_pinecone(self, file_id, file_path=None, file_structure=None, content=None):
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
            
            # Process each text chunk with key rotation if needed
            successful_chunks = 0
            failed_chunks = 0
            
            for i, text in enumerate(texts):
                try:
                    print(f"Processing chunk {i+1}/{len(texts)} ({len(text)} chars)")
                    vector = self.embed_text(text, file_path, file_structure)
                    vectors.append(vector)
                    successful_chunks += 1
                    
                    # Add a small delay between chunks to prevent rate limiting
                    if i < len(texts) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    failed_chunks += 1
                    if "All Cohere API keys have been exhausted" in str(e) or "All Mistral API keys have been exhausted" in str(e):
                        print(f"Failed to embed chunk: {e}")
                        # If we're out of API keys, stop processing
                        break
                    print(f"Error embedding chunk {i+1}: {e}")
                    # Continue with next chunk if possible
            
            print(f"Processing complete: {successful_chunks} successful, {failed_chunks} failed chunks")
            
            if not vectors:
                raise Exception("No vectors were successfully created")

            # Upload vectors to Pinecone
            print(f"Uploading {len(vectors)} vectors to Pinecone namespace: {namespace_name}")
            index = self.pc.Index(self.index_name)
            response = index.upsert(vectors=vectors, namespace=namespace_name)

            print(f"Successfully uploaded vectors to Pinecone: {response}")
            return namespace_name
        
        except Exception as error:
            print("Error in load_text_to_pinecone:", error)
            raise error

    def retrieve_data_from_pincone(self, context, namespace=None):
        index = self.pc.Index(self.index_name)
        print(f"Context to retrieve: {context[:50]}..." if len(context) > 50 else context)
        
        # Always use Cohere for retrieval
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
                    "search_kwargs": {"k": 3, "score_threshold": 0.1}
                }
                
                if namespace:
                    retriever_kwargs["search_kwargs"]["namespace"] = namespace
                
                retriever = vector_store.as_retriever(**retriever_kwargs)
                
                result = retriever.invoke(context)
                return result
                
            except Exception as error:
                error_str = str(error).lower()
                print(f"Retrieval error: {error_str}")
                
                if "rate" in error_str or "limit" in error_str or "quota" in error_str or error_str == "'data'" or "429" in error_str:
                    if self.current_cohere_key_index + 1 < len(self.cohere_keys):
                        print(f"Cohere API key #{self.current_cohere_key_index + 1} exhausted during retrieval. Trying next key...")
                        self._rotate_cohere_key()
                    else:
                        print(f"All Cohere keys exhausted during retrieval.")
                        retry_count = max_retries  # Force exit loop
                
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed retrieval after {max_retries} attempts: {error}")
                    raise Exception(f"Failed to retrieve data after {max_retries} attempts: {error}")
                
                # Longer delay before retry
                time.sleep(2)
        
        raise Exception("Unexpected error in retrieval function")

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
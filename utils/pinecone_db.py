from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from hashlib import md5
from pinecone import Pinecone, ServerlessSpec
import os
import re
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

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
            os.getenv("MISTRAL_API_KEY_1"),
            os.getenv("MISTRAL_API_KEY_2"),
            os.getenv("MISTRAL_API_KEY_3")
        ]
        self.current_mistral_key_index = 0
        
        # Default to Cohere embeddings initially
        self.current_embedding_type = "cohere"
        self.embeddings = self._get_cohere_embeddings()
        self.index_name = index
        
        # Code file extensions to use Mistral Codestral for
        self.code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.html', '.css', '.ts', '.sh', '.json', '.xml', '.yaml', '.yml', '.sql', '.rs', '.swift', '.kt']

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
            model="codestral-latest"
        )
    
    def _is_code_file(self, file_path):
        """Check if the file is a source code file based on extension"""
        if not file_path:
            return False
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.code_extensions
    
    def _select_embeddings_model(self, file_path=None):
        """Select appropriate embedding model based on file type"""
        if self._is_code_file(file_path):
            if self.current_embedding_type != "mistral":
                self.current_embedding_type = "mistral"
                self.embeddings = self._get_mistral_embeddings()
                print("Using Mistral Codestral model for code file")
        else:
            if self.current_embedding_type != "cohere":
                self.current_embedding_type = "cohere"
                self.embeddings = self._get_cohere_embeddings()
                print("Using Cohere model for text file")
        
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
        # Select appropriate model based on file type
        self._select_embeddings_model(file_path)
        
        if self.current_embedding_type == "cohere":
            keys_tried = 0
            while keys_tried < len(self.cohere_keys):
                try:
                    res = self.embeddings.embed_documents([text])
                    return res
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate" in error_str or "limit" in error_str or "quota" in error_str:
                        print(f"Cohere API key #{self.current_cohere_key_index + 1} exhausted. Trying next key...")
                        self._rotate_cohere_key()
                        keys_tried += 1
                    else:
                        print("Error calling Cohere embeddings API:", error)
                        raise error
            
            # If all Cohere keys are exhausted
            raise Exception("All Cohere API keys have been exhausted")
        
        else:  # Mistral embeddings
            keys_tried = 0
            while keys_tried < len(self.mistral_keys):
                try:
                    res = self.embeddings.embed_documents([text])
                    return res
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate" in error_str or "limit" in error_str or "quota" in error_str:
                        print(f"Mistral API key #{self.current_mistral_key_index + 1} exhausted. Trying next key...")
                        self._rotate_mistral_key()
                        keys_tried += 1
                    else:
                        print("Error calling Mistral embeddings API:", error)
                        raise error
            
            # If all Mistral keys are exhausted
            raise Exception("All Mistral API keys have been exhausted")

    def embed_text(self, text, file_path=None, file_structure=None):
        try: 
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
            print("Error embedding text:", error)
            raise error

    def truncate_string_by_bytes(self, string, max_bytes):
        encoded = string.encode("utf-8")
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def prepare_document(self, content, chunk_size=800, chunk_overlap=50):
        try:
            content = content.replace("\n", "")
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_text(content)
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
            
        if file_path and not content:
            with open(file_path, 'r') as f:
                content = f.read()
        
        namespace_name = file_id
        try:
            texts = self.prepare_document(content)
            vectors = []
            
            # Process each text chunk with key rotation if needed
            for text in texts:
                try:
                    vector = self.embed_text(text, file_path, file_structure)
                    vectors.append(vector)
                except Exception as e:
                    if "All Cohere API keys have been exhausted" in str(e) or "All Mistral API keys have been exhausted" in str(e):
                        raise e
                    print(f"Error embedding chunk: {e}")
                    # Continue with next chunk if possible
            
            if not vectors:
                raise Exception("No vectors were successfully created")

            # Upload vectors to Pinecone
            index = self.pc.Index(self.index_name)
            response = index.upsert(vectors=vectors, namespace=namespace_name)

            print("Successfully uploaded vectors to Pinecone")
            return namespace_name
        
        except Exception as error:
            print("Error in load_text_to_pinecone:", error)
            raise error

    def retrieve_data_from_pincone(self, context, current_embedding_type=None):
        index = self.pc.Index(self.index_name)
        print("Context to retrieve for ", context)
        
        # Select appropriate model based on the query context or file path
        # self._select_embeddings_model(file_path)
        
        # Try each key for retrieval
        if current_embedding_type == "cohere":
            keys_tried = 0
            while keys_tried < len(self.cohere_keys):
                try:
                    vector_store = PineconeVectorStore(
                        index=index,  
                        embedding=self.embeddings
                    )

                    retriever = vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": 3, "score_threshold": 0.1},
                    )
                    
                    result = retriever.invoke(context)
                    return result
                    
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate" in error_str or "limit" in error_str or "quota" in error_str:
                        print(f"Cohere API key #{self.current_cohere_key_index + 1} exhausted during retrieval. Trying next key...")
                        self._rotate_cohere_key()
                        keys_tried += 1
                    else:
                        print("Error during retrieval:", error)
                        raise error
            
            # If all keys are exhausted
            raise Exception("All Cohere API keys have been exhausted during retrieval")
        
        else:  # Mistral embeddings
            keys_tried = 0
            while keys_tried < len(self.mistral_keys):
                try:
                    vector_store = PineconeVectorStore(
                        index=index,  
                        embedding=self.embeddings
                    )

                    retriever = vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": 3, "score_threshold": 0.1},
                    )
                    
                    result = retriever.invoke(context)
                    return result
                    
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate" in error_str or "limit" in error_str or "quota" in error_str:
                        print(f"Mistral API key #{self.current_mistral_key_index + 1} exhausted during retrieval. Trying next key...")
                        self._rotate_mistral_key()
                        keys_tried += 1
                    else:
                        print("Error during retrieval:", error)
                        raise error
            
            # If all keys are exhausted
            raise Exception("All Mistral API keys have been exhausted during retrieval")

    def delete_index(self):
        if self.pc.has_index(self.index_name):
            self.pc.delete_index(self.index_name)
            print("Index deleted successfully")

pinecone = pineconeOperation('repository')
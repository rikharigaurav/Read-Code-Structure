from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone
# from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere import CohereEmbeddings
from hashlib import md5
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
    
class pineconeOperation:
    def __init__(self, index):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY_1'),model="embed-english-v3.0")
        self.index_name = index

    def convert_to_serializable(self, data):
        """Convert non-serializable tuple keys to strings."""
        if isinstance(data, dict):
            return {str(k): self.convert_to_serializable(v) for k, v in data.items()}
        return data

    def get_embeddings(self, text: str):
        try:
            res = self.embeddings.embed_documents([text])
            # print(res)
            return res
        except Exception as error:
            print("Error calling embeddings API", error)
            raise error
        

    def embed_text(self, text, file_structure=None):
        try: 
            # print("Text being processed:", text)

            embeddings = self.get_embeddings(text)
            # print("Embeddings:", embeddings)

            hash_id = md5(text.encode()).hexdigest()
            # print("Hash:", hash_id)
            metadata = {
                "text": self.truncate_string_by_bytes(text, 3600)
            }
            
            if file_structure is not None:
                # serializable_structure = {}
                # for key, value in file_structure.items():
                #     if isinstance(key, tuple):
                #         new_key = f"{key[0]}:{key[1]}"  
                #     else:
                #         new_key = str(key)
                    
                #     # Also convert any nested dictionaries
                #     if isinstance(value, dict):
                #         serializable_structure[new_key] = self.convert_to_serializable(value)
                #     else:
                #         serializable_structure[new_key] = value
                
                metadata["File_Structure"] = file_structure

            record = {
                "id": hash_id,
                "values": embeddings[0],    
                "metadata": metadata,
            }
            # print("Pinecone Record:", record)

            return record
        except Exception as error:
            print("Error embedding text:", error)
            raise error

    # Truncate a string to fit a specific byte size
    def truncate_string_by_bytes(self, string, max_bytes):
        encoded = string.encode("utf-8")
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    # Prepare a document by splitting a string
    def prepare_document(self, content, chunk_size=800,chunk_overlap=50):
        try:
            content = content.replace("\n", "")
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            docs = splitter.split_text(content)
            return docs
        except Exception as error:
            print("Error preparing document:", error)
            raise error

    # Load text data to Pinecone and return the namespace name
    def load_text_to_pinecone(self, file_path, file_id,file_structure=None):
        content = None
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

        with open(file_path, 'r') as f:
            content = f.read()
        
        namespace_name = file_id
        try:
            # print("Processing content for namespace:", namespace_name)

            # Split and embed text
            # serializable_structure = self.convert_to_serializable(file_structure) if file_structure else None
            texts = self.prepare_document(content)
            vectors = [self.embed_text(text, file_structure) for text in texts]
            # print("Vectors of the text:", vectors)

            # Upload vectors to Pinecone
            index = self.pc.Index(self.index_name)
            response = index.upsert(vectors=vectors, namespace=namespace_name)
            # print("Vectors uploaded to Pinecone:", response)

            print("Successfully uploaded vectors to Pinecone")
            return namespace_name
        
        except Exception as error:
            print("Error in load_text_to_pinecone:", error)
            raise error


    def retrieve_data_from_pincone(self, context):
        index = self.pc.Index(self.index_name)
        print("Context to retrieve for ", context)
        vector_store = PineconeVectorStore(
            index=index,  
            embedding=self.embeddings
        )

        # Create and use the retriever
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        
        result = retriever.invoke(context)
        
        
        return result

    def delete_index(self):
        if self.pc.has_index(self.index_name):
            self.pc.delete_index(self.index_name)
            print("Index deleted successfully")

pinecone = pineconeOperation('repository')
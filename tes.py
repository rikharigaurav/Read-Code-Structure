from utils.pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import CohereEmbeddings
from hashlib import md5
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = CohereEmbeddings(
    api_key=os.getenv("COHERE_API_KEY"),
    batch_size=48
)


async def get_embeddings(text: str):
    try:
        res = embeddings.embed_query(text)
        print(res)
        return res
    except Exception as error:
        print("Error calling embeddings API", error)
        raise error
    
# Embed a single document
def embed_text(text):
    try:
        print("Text being processed:", text)

        # Get embeddings
        embeddings = get_embeddings(text)
        print("Embeddings:", embeddings)

        # Create a hash ID
        hash_id = md5(text.encode()).hexdigest()
        print("Hash:", hash_id)

        # Prepare Pinecone record
        record = {
            "id": hash_id,
            "values": embeddings,
            "metadata": {
                "text": truncate_string_by_bytes(text, 3600),
            },
        }
        print("Pinecone Record:", record)

        return record
    except Exception as error:
        print("Error embedding text:", error)
        raise error

# Truncate a string to fit a specific byte size
def truncate_string_by_bytes(string, max_bytes):
    encoded = string.encode("utf-8")
    return encoded[:max_bytes].decode("utf-8", errors="ignore")

# Prepare a document by splitting a string
def prepare_document(content, chunk_size=800,chunk_overlap=50):
    try:
        content = content.replace("\n", "")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        docs = splitter.split_text(content)
        return docs
    except Exception as error:
        print("Error preparing document:", error)
        raise error

# Load text data to Pinecone and return the namespace name
def load_text_to_pinecone(file_path, index_name):
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        )
    
    namespace_name = ""
    try:
        print("Processing content for namespace:", namespace_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split and embed text
        texts = prepare_document(content)
        vectors = [embed_text(text) for text in texts]
        print("Vectors of the text:", vectors)

        # Upload vectors to Pinecone
        response = pc.upsert(vectors=vectors, namespace=namespace_name)
        print("Vectors uploaded to Pinecone:", response)

        print("Successfully uploaded vectors to Pinecone")
        return namespace_name
    except Exception as error:
        print("Error in load_text_to_pinecone:", error)
        raise error

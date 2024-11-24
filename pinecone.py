from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, dotenv_values
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "example-index"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
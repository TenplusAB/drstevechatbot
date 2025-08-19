import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()
OPEN_AI_API = os.getenv("OPEN_AI_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI client
client = OpenAI(api_key=OPEN_AI_API)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)  # existing index

# Load your JSON file
with open("analyzed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Embed and upload
for i, item in enumerate(data):
    answer_text = item["answer"]
    diseases = item.get("diseases", [])
    keywords = item.get("keywords", [])

    # Create embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=answer_text
    )
    embedding_vector = embedding_response.data[0].embedding

    # Metadata
    metadata = {
        "diseases": diseases,
        "keywords": keywords,
        "answer": answer_text
    }

    # Upload to Pinecone
    index.upsert([
        (f"doc-{i}", embedding_vector, metadata)
    ])

print("✅ All data embedded and uploaded to Pinecone.")

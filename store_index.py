import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medsage-openai"
MAX_CHUNKS = os.getenv("MAX_CHUNKS")
MAX_CHUNKS = int(MAX_CHUNKS) if MAX_CHUNKS and MAX_CHUNKS.isdigit() else None

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Dimension for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")

# Load documents
print("Loading PDFs from Data/ ...")
extracted_data = load_pdf_file("Data/")
print(f"Loaded {len(extracted_data)} pages/documents")
text_chunks = text_split(extracted_data)
print(f"Split into {len(text_chunks)} chunks")
if MAX_CHUNKS is not None:
    text_chunks = text_chunks[:MAX_CHUNKS]
    print(f"Limiting to MAX_CHUNKS={MAX_CHUNKS} chunks for indexing")

# Load embeddings
print("Initializing embeddings ...")
embeddings = download_hugging_face_embeddings()

# Store in Pinecone (disable multiprocessing on Windows)
print("Upserting embeddings into Pinecone (this can take a while) ...")
PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    pool_threads=1,
)

print("✅ Index stored successfully in Pinecone.")


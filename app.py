import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.helper import download_hugging_face_embeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Pinecone index name for MedSage (OpenAI embeddings: 1536 dims)
INDEX_NAME = "medsage-openai"

# Initialize Flask
app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Connect to vector store
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500
)

# System prompt
system_prompt = """
You are MedSage, a medical assistant chatbot.
Answer the question using the provided context.
If you don't know the answer, say you don't know.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    response = rag_chain.invoke({"input": user_input})
    return response["answer"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
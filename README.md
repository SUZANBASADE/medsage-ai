🩺 MedSage (Medical RAG Chatbot) — LangChain + Pinecone

This project is a medical question-answering chatbot built using LangChain and Pinecone.
It reads medical PDF files and answers user questions based on those documents.

The chatbot uses Retrieval-Augmented Generation (RAG), which means:

It first searches relevant content from PDFs

Then uses an LLM to generate answers

This helps reduce wrong or made-up answers

🛠 Tech Stack

Python

Flask – backend web framework

LangChain – RAG pipeline

Pinecone – vector database

HuggingFace Embeddings – text embeddings

OpenAI – LLM for answering questions

📂 Project Structure

MedSage/
│
├── app.py                  # Flask app (chat API)
├── store_index.py          # Script to index PDFs into Pinecone
├── src/
│   ├── helper.py           # PDF loading, splitting, embeddings
│   └── prompt.py           # System prompt
│
├── templates/
│   └── chat.html           # Chat UI
│
├── Data/                   # Medical PDF files
│   └── *.pdf
│
├── requirements.txt
├── .env                    # Environment variables (NOT pushed to GitHub)
└── README.md


🔄 How the Project Works

Medical PDFs are loaded from the Data/ folder

Text is split into small chunks

Each chunk is converted into embeddings

Embeddings are stored in Pinecone

User asks a question

Relevant text is retrieved from Pinecone

LLM generates a short, accurate answer

⚙️ Setup Instructions (Step by Step)
1️⃣ Clone the repository 👉https://github.com/SUZANBASADE/medsage-ai/upload/main
Download / clone this project, then `cd` into the folder.

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🔐 Pinecone Setup (Important)
Step 1: Create Pinecone Account

Go to 👉 https://www.pinecone.io

Sign up / log in

Go to Dashboard

Step 2: Create Pinecone API Key

Open API Keys section

Copy your API Key

Step 3: Create .env file

In the project root, create a file named .env

PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here


📌 Index PDFs into Pinecone

Run this once to store embeddings:

python store_index.py


This will:

Read PDFs

Create embeddings

Upload them to Pinecone

🚀 Run the Application
python app.py


Open browser:

http://localhost:5000


Ask medical questions through the chat UI.

🧪 Example Questions

What is diabetes?

What are the symptoms of heart disease?

How is hypertension treated?

⚠️ Current Limitations

Single-turn questions only (no chat memory)

No PDF upload from UI

Uses similarity-based retrieval only

🔮 Future Improvements

Conversational memory

Source citations

Metadata filtering

Streaming responses

Authentication & rate limiting

📌 Notes

This project is meant for learning and demo purposes

Not intended for real medical diagnosis

Answers depend on uploaded PDFs

👨‍💻 Author

Suzan Basade
Computer Science | Data Analytics Enthusiast

⭐ If you like this project

Give it a ⭐ on GitHub 😊



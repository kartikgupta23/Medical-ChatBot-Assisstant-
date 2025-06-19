# Importing Libraries
from langchain_community.vectorstores import FAISS  # << FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from src.helper import load_pdf, text_split, download_huggingface_embedding_model

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Loading environment variables...")
print("HUGGINGFACEHUB_API_TOKEN:", os.getenv("HF_TOKEN"))

# Load the data
print("📚 Loading PDF documents...")
extracted_data = load_pdf("./data")

# Split the documents
print("✂️ Splitting documents into chunks...")
text_chunks = text_split(extracted_data)

# Download embedding model
print("⬇️ Downloading Hugging Face embedding model...")
embeddings = download_huggingface_embedding_model()

# Create embeddings
print("🧠 Creating embeddings...")
texts = [t.page_content for t in text_chunks]

# FAISS: create vectorstore from documents
print("⚡ Building FAISS vector store locally...")
vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)

# Save the FAISS index locally
vectorstore.save_local("faiss_index")

print("✅ FAISS Vectorstore created and saved successfully!")


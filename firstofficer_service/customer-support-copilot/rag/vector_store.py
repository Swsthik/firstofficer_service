import os
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

# Directory containing documents
DATA_DIR = "data"
VECTOR_STORE_PATH = "rag/vector_store/faiss_index"

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

def build_vector_store():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")

    documents = []

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)

        documents.extend(loader.load())

    if not documents:
        raise ValueError("No documents found to build the vector store.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks. Creating embeddings...")

    # Build FAISS vector store
    db = FAISS.from_documents(docs, embedding)

    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Removing existing vector store at {VECTOR_STORE_PATH}...")
        shutil.rmtree(VECTOR_STORE_PATH)

    # Save to disk
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    db.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store saved to {VECTOR_STORE_PATH}.")

if __name__ == "__main__":
    build_vector_store()

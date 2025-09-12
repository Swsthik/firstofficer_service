import os
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Directory containing documents
DATA_DIR = "data/secure-agent"
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
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            loader = UnstructuredFileLoader(file_path)

        # Each loaded doc gets metadata automatically
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata.update({
                "filename": filename,
                # placeholder for future URL mapping
                "source": f"local://{filename}"
            })
        documents.extend(loaded_docs)

    if not documents:
        raise ValueError("No documents found to build the vector store.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks. Creating embeddings...")

    # If FAISS index already exists, load & append
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
        db = FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        print("Creating new vector store...")
        db = FAISS.from_documents(docs, embedding)

    # Save updated DB
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    db.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store updated and saved to {VECTOR_STORE_PATH}.")

if __name__ == "__main__":
    build_vector_store()

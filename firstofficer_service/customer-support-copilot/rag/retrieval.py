from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()

# --- Vector store config ---
VECTOR_STORE_PATH = "rag/rag/vector_store/faiss_index"

# Embeddings model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# --- Load FAISS index ONCE at startup ---
try:
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    db = None
    print(f"⚠️ Failed to load FAISS index from {VECTOR_STORE_PATH}: {e}")

# --- Core Retrieval Function ---
def retrieve_and_answer(query, k=3):
    if db is None:
        return "⚠️ Vector database not available. Please check FAISS index path."

    # Retrieve top-k relevant docs
    results = db.similarity_search(query, k=k)

    if not results:
        return (
            "No relevant information found in the documentation. "
            "This ticket should be routed to the support team."
        )

    # Build response with sources
    context_parts = []
    sources = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown Source")
        context_parts.append(f"{doc.page_content}")
        sources.append(source)

    answer = (
        f"Answer: {' '.join(context_parts)}\n"
        f"Sources: {', '.join(set(sources))}"
    )
    return answer


if __name__ == "__main__":
    query = "What are the system requirements for installing the virtual machine?"
    answer = retrieve_and_answer(query, k=3)

    print("🔎 Answer:")
    print(answer)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

VECTOR_STORE_PATH = "rag/vector_store/faiss_index"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

assistant_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an AI assistant. Use the provided context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer concisely and clearly. If the context does not contain enough information, say so."
    ),
)

def retrieve_and_answer(query, k=3):
    # Load FAISS index
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    # Retrieve top-k relevant docs
    results = db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])

    # Format the assistant prompt
    formatted_prompt = assistant_prompt.format(context=context, query=query)

    # Get LLM response
    response = llm.invoke(formatted_prompt)

    return response.content


if __name__ == "__main__":
    query = "What are the steps to modify a playbook?"
    answer = retrieve_and_answer(query, k=3)

    print("🔎 Answer:")
    print(answer)

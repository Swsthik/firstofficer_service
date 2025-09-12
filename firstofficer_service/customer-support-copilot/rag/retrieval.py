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

from langchain.prompts import PromptTemplate

assistant_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are a Customer Support Copilot for a software product. "
        "Your role is to assist the support team by answering customer queries using ONLY the provided context. "
        "Follow these rules strictly:\n"
        "1. Use the context to generate a clear and concise answer to the customer's question.\n"
        "2. If the context does not provide enough information, respond with: "
        "'No relevant information found in the documentation. This ticket should be routed to the support team.'\n"
        "3. Always include the sources (URLs) from the context that support your answer.\n"
        "4. Never invent information or cite non-existent sources.\n\n"

        "Context (with sources):\n{context}\n\n"
        "Customer Query: {query}\n\n"

        "Provide your response in this format:\n"
        "Answer: <clear and concise answer to the customer>\n"
        "Sources: <comma-separated list of URLs used>\n"
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
    query = "What are the system requirements for installing the virtual machine?"
    answer = retrieve_and_answer(query, k=3)

    print("🔎 Answer:")
    print(answer)

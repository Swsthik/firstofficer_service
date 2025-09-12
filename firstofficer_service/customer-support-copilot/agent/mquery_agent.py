import os
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieval  # your retrieval.py

# Initialize LLM
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.3
)

# Topics that require RAG
RAG_TOPICS = ["How-to", "Product", "API/SDK", "SSO", "Best practices"]

# A simple context-aware agent
class MultiQueryAgent:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history"""
        self.history.append({"role": role, "content": content})

    def generate_response(self, user_input: str) -> str:
        """
        Generate response to a user query.
        - If first input is casual greeting, respond conversationally.
        - If user asks a technical/support question, optionally call RAG.
        """
        # Add user input to history
        self.add_to_history("user", user_input)

        # Combine previous conversation for context
        context_text = ""
        for msg in self.history[-6:]:  # keep last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content']}\n"

        # Simple check: greetings or casual queries
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if len(self.history) == 1 and user_input.lower().strip() in greetings:
            reply = f"{user_input.capitalize()}! How can I help you today?"
            self.add_to_history("assistant", reply)
            return reply

        # Otherwise, check if the query should trigger RAG
        prompt = f"""
You are a Customer Support Copilot. Given the following conversation, decide whether the user query requires knowledge from documentation or developer resources.
Respond with either "RAG" or "NO_RAG".

Conversation:
{context_text}

Decision:
"""
        decision_resp = llm.invoke(prompt).content.strip().upper()
        if "RAG" in decision_resp:
            # Call retrieval agent
            answer = retrieval.retrieve_and_answer(user_input)
        else:
            # Provide a conversational fallback
            fallback_prompt = f"""
You are a helpful AI assistant. Continue the conversation with the user based on the following context:
{context_text}
Respond conversationally and politely.
"""
            answer = llm.invoke(fallback_prompt).content.strip()

        # Add assistant response to history
        self.add_to_history("assistant", answer)
        return answer


# ---- Wrapper for app.py ----
_agent_instance = MultiQueryAgent()

def handle_message(user_query, retrieval, classifier_agent=None, RAG_TOPICS=None):
    """
    Wrapper for app.py compatibility.
    Uses MultiQueryAgent.generate_response() internally.
    """
    return _agent_instance.generate_response(user_query)


# Example standalone usage
if __name__ == "__main__":
    agent = MultiQueryAgent()

    print(agent.generate_response("hi"))
    print(agent.generate_response("What are the system requirements for installing the virtual machine?"))
    print(agent.generate_response("Do I need admin rights for this installation?"))

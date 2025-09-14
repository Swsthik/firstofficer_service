import os
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import ticket_agent
from agent import rag_agent  # Full RAGAgent
from dotenv import load_dotenv

# --- Load environment and initialize LLM ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.3
)

# Initialize a single RAGAgent instance
_rag_agent_instance = rag_agent.RAGAgent()


class MultiQueryAgent:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.last_log: Dict = {}

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def generate_response(self, user_input: str) -> str:
        self.add_to_history("user", user_input)

        # Build last 6 messages for context
        context_text = ""
        for msg in self.history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content']}\n"

        # --- Greeting / casual reply ---
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if len(self.history) == 1 and user_input.lower().strip() in greetings:
            answer = f"{user_input.capitalize()}! How can I help you today?"
            log = {
                "Type": "ai_response",
                "Content": answer,
                "Classification": "N/A",
                "Escalation Score": 0.0,
                "Factors": {},
                "Reasoning": [],
                "Sources": [],
                "Ticket ID": None,
                "Should Escalate": False
            }
            self.add_to_history("assistant", answer)
            self.last_log = log
            print("\nLog:", log)
            return answer

        # --- Decide if RAG is needed ---
        decision_prompt = f"""
        You are a Customer Support Copilot. Given the following conversation, decide whether the user query requires knowledge from documentation or developer resources.
        Respond with either "RAG" or "NO_RAG".

        Conversation:
        {context_text}

        Decision:
        """
        decision_resp = llm.invoke(decision_prompt).content.strip().upper()

        factors = {}
        reasoning = []
        should_escalate = False
        ticket_id = None
        sources = []

        if "RAG" in decision_resp:
            # Call full RAGAgent
            rag_result = _rag_agent_instance.process_query(user_input)

            # Escalation based on RAG factors
            factors = rag_result.get("escalation", {}).get("factors", {})
            if factors.get("sentiment_urgency", 0) >= 0.6:
                should_escalate = True
                reasoning.append("High urgency detected")
            if factors.get("topic_criticality", 0) >= 0.6:
                should_escalate = True
                reasoning.append("Critical topic")

            ticket_id = ticket_agent.create_ticket(
                query=user_input,
                classification=rag_result.get("classification", {}),
                response=rag_result.get("draft_answer", ""),
                escalation_info=rag_result.get("escalation", {}),
            )

            # LLM generates final answer using retrieved content
            combined_prompt = f"""
            You are a helpful AI assistant.
            - Answer the user's query using the retrieved content.
            - Conversation context is provided below.
            - Escalation Required: {should_escalate}
            - Escalation Reasoning: {" | ".join(reasoning) if reasoning else "N/A"}

            Retrieved Content:
            {rag_result.get("draft_answer", "No relevant content found")}

            Conversation Context:
            {context_text}

            User Query:
            {user_input}

            Respond conversationally and helpfully.
            - If Escalation Required is True, politely inform the user that their query has also been routed to our support team.
            - Include all relevant information from the retrieved content in your response.
            """
            answer = llm.invoke(combined_prompt).content.strip()
            sources = rag_result.get("sources", [])
            log_type = "ai_escalation" if should_escalate else "ai_response"

        else:
            # Normal fallback
            fallback_prompt = f"""
            You are a helpful AI assistant. Continue the conversation with the user based on the following context:
            {context_text}
            Respond conversationally and politely.
            """
            answer = llm.invoke(fallback_prompt).content.strip()
            log_type = "ai_response"

        # --- Response Quality Check on final LLM answer ---
        response_quality = 0.7
        low_quality_markers = [
            "no relevant content",
            "not enough information",
            "cannot find",
            "no information available",
            "unknown",
            "not contain information",
            "I couldn't find any information",
        ]
        if any(marker in answer.lower() for marker in low_quality_markers):
            response_quality = 0.5
            should_escalate = True
            reasoning.append("Low answer quality")

        factors["response_quality"] = response_quality

        # Build structured log
        log = {
            "Type": log_type,
            "Content": answer,
            "Classification": rag_result.get("classification", {}) if "rag_result" in locals() else "N/A",
            "Escalation Score": rag_result.get("escalation", {}).get("escalation_score", 0.0) if "rag_result" in locals() else 0.0,
            "Factors": factors,
            "Reasoning": reasoning,
            "Sources": sources,
            "Ticket ID": ticket_id,
            "Should Escalate": should_escalate,
        }

        self.add_to_history("assistant", answer)
        self.last_log = log
        print("\nLog:", log)

        # Append sources to answer for UI
        if sources:
            answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

        return answer


# ---- Wrapper for app.py ----
_agent_instance = MultiQueryAgent()


def handle_message(user_query, return_log=False):
    response = _agent_instance.generate_response(user_query)
    if return_log:
        return response, getattr(_agent_instance, "last_log", None)
    return response


# Example standalone usage
if __name__ == "__main__":
    agent = MultiQueryAgent()
    print(agent.generate_response("hi"))
    print(agent.generate_response("What are the system requirements for installing the virtual machine?"))
    print(agent.generate_response("Do I need admin rights for this installation?"))
    print(agent.generate_response("This is extremely urgent! I need it immediately!"))

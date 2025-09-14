# rag_agent.py
import os
from typing import Dict
from dotenv import load_dotenv
from agent.classifier_agent import classify_ticket
from rag import retrieval

load_dotenv()


class EscalationDecisionEngine:
    def __init__(self, escalation_threshold: float = None):
        if escalation_threshold is None:
            escalation_threshold = float(os.getenv("ESCALATION_THRESHOLD", 0.6))
        self.escalation_threshold = escalation_threshold

    def score(self, query: str, classification: Dict, draft_answer: str = None) -> Dict:
        """Return escalation decision + reasoning (no text response)."""

        # --- Query complexity ---
        complexity = 0.0
        if "?" in query:
            complexity += 0.3
        if len(query.split()) > 50:
            complexity += 0.2
        if any(word in query.lower() for word in ["integrate", "custom", "setup", "troubleshoot"]):
            complexity += 0.2

        # --- Sentiment/urgency ---
        sentiment = classification.get("sentiment", "Neutral")
        priority = classification.get("priority", "P2")
        sentiment_urgency = 0.0
        if sentiment == "Angry":
            sentiment_urgency += 0.9
        elif sentiment == "Frustrated":
            sentiment_urgency += 0.7
        if priority == "P0":
            sentiment_urgency += 0.8
        elif priority == "P1":
            sentiment_urgency += 0.4

        # --- Topic criticality ---
        topic = classification.get("topic", "Unknown")
        critical_topics = {
            "Sensitive data": 0.9,
            "Security": 0.8,
            "Billing": 0.7,
            "Compliance": 0.8,
            "Integration": 0.6,
            "Custom": 0.7,
        }
        topic_criticality = critical_topics.get(topic, 0.2)

        # --- Response quality ---
        response_quality = 0.0

        # --- Weighted score ---
        weights = {
            "complexity": 0.3,
            "sentiment_urgency": 0.3,
            "topic_criticality": 0.3,
            "response_quality": 0.1,
        }
        factors = {
            "complexity": complexity,
            "sentiment_urgency": sentiment_urgency,
            "topic_criticality": topic_criticality,
            "response_quality": response_quality,
        }
        escalation_score = sum(factors[k] * weights[k] for k in weights)
        should_escalate = escalation_score > self.escalation_threshold

        reasoning = []
        if factors["complexity"] > 0.3:
            reasoning.append("Complex query")
        if factors["sentiment_urgency"] > 0.7:
            reasoning.append("High urgency/sentiment")
        if factors["topic_criticality"] > 0.7:
            reasoning.append("Critical topic")
        if factors["response_quality"] > 0.3:
            reasoning.append("Low answer quality")

        return {
            "should_escalate": should_escalate,
            "escalation_score": escalation_score,
            "factors": factors,
            "reasoning": reasoning,
        }


class RAGAgent:
    def __init__(self, escalation_threshold: float = None):
        self.escalation_engine = EscalationDecisionEngine(escalation_threshold)

    def process_query(self, query: str) -> Dict:
        """
        Classify, retrieve draft, and score escalation.
        Does NOT produce final LLM response.
        """
        # --- Classify query ---
        classification = classify_ticket(query)

        # --- Retrieval ---
        draft_answer = retrieval.retrieve_and_answer(query, k=3)  # Removed 'return_sources'

        # --- Escalation decision ---
        decision = self.escalation_engine.score(query, classification, draft_answer)

        return {
            "classification": classification,
            "draft_answer": draft_answer,
            "escalation": decision,
        }


# Example usage
if __name__ == "__main__":
    agent = RAGAgent()
    test_queries = [
        "How do I set up SSO authentication with SAML?",
        "I'm really frustrated! Your billing system charged me twice!",
        "What are the system requirements for installing the virtual machine?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = agent.process_query(q)
        print("Draft Answer:", result["draft_answer"])
        print("Classification:", result["classification"])
        print("Escalation:", result["escalation"])
    

import time
import random
import string
from typing import Dict, List, Optional
from datetime import datetime
from agent.mquery_agent import _agent_instance as mquery_agent
from agent.rag_agent import RAGAgent
from agent.classifier_agent import classify_ticket

class TicketAgent:
    def __init__(self):
        self.tickets: Dict[str, Dict] = {}
        self.ticket_counter = 0
        self.rag_agent = RAGAgent()

    def generate_ticket_id(self) -> str:
        """
        Generate a unique ticket ID using timestamp and random components.
        Algorithm: TICK-{timestamp}-{counter}-{random_suffix}
        """
        timestamp = int(time.time() * 1000)  # milliseconds
        self.ticket_counter += 1
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        ticket_id = f"TICK-{timestamp}-{self.ticket_counter:04d}-{random_suffix}"
        return ticket_id

    def process_query_with_ticket(self, query: str) -> Dict:
        """
        Process query through mquery agent, and create ticket if RAG is used or escalation is needed.
        Returns response dict with ticket info if created.
        """
        # Get response from mquery agent
        response = mquery_agent.generate_response(query)
        
        # Check if RAG was used by detecting the formatted response structure
        if "Answer:" in response and "Sources:" in response:
            # RAG was used, get full analysis from RAG agent
            rag_result = self.rag_agent.process_query(query)
            
            # Create ticket for RAG queries (including escalations)
            ticket_id = self.create_ticket(
                query,
                rag_result['classification'],
                rag_result['content'],
                {
                    'escalation_score': rag_result['escalation_score'],
                    'factors': rag_result['factors'],
                    'reasoning': rag_result['reasoning'],
                    'type': rag_result['type']
                }
            )
            
            return {
                "response": rag_result['content'],
                "ticket_id": ticket_id,
                "type": rag_result['type'],
                "classification": rag_result['classification'],
                "escalation_score": rag_result['escalation_score'],
                "factors": rag_result['factors'],
                "reasoning": rag_result['reasoning']
            }
        else:
            # Conversational response, check for escalation indicators
            if "routed to the support team" in response.lower() or "escalate" in response.lower():
                # Escalation needed, classify and create ticket
                classification = classify_ticket(query)
                
                ticket_id = self.create_ticket(
                    query,
                    classification,
                    response,
                    {
                        'type': 'escalation',
                        'reasoning': ['Fallback escalation from conversational response']
                    }
                )
                
                return {
                    "response": response,
                    "ticket_id": ticket_id,
                    "type": "escalation",
                    "classification": classification
                }
            else:
                # Pure conversational response, no ticket
                return {
                    "response": response,
                    "ticket_id": None,
                    "type": "conversational"
                }

    def create_ticket(self, query: str, classification: Dict, response: str,
                     escalation_info: Optional[Dict] = None) -> str:
        """
        Create a new ticket with associated data.
        """
        ticket_id = self.generate_ticket_id()

        ticket_data = {
            "ticket_id": ticket_id,
            "query": query,
            "classification": classification,
            "response": response,
            "escalation_info": escalation_info,
            "created_at": datetime.now().isoformat(),
            "status": "open",  # open, escalated, closed
            "priority": classification.get("priority", "P2"),
            "topic": classification.get("topic", "Unknown"),
            "sentiment": classification.get("sentiment", "Neutral")
        }

        self.tickets[ticket_id] = ticket_data
        return ticket_id

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Retrieve a ticket by ID"""
        return self.tickets.get(ticket_id)

    def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        """Update ticket status (open, escalated, closed)"""
        if ticket_id in self.tickets:
            self.tickets[ticket_id]["status"] = status
            self.tickets[ticket_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False

    def list_tickets(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """List tickets, optionally filtered by status"""
        tickets = list(self.tickets.values())
        if status:
            tickets = [t for t in tickets if t["status"] == status]
        # Sort by creation time, newest first
        tickets.sort(key=lambda x: x["created_at"], reverse=True)
        return tickets[:limit]

    def get_ticket_stats(self) -> Dict:
        """Get statistics about tickets"""
        total = len(self.tickets)
        by_status = {}
        by_priority = {}
        by_topic = {}
        by_sentiment = {}

        for ticket in self.tickets.values():
            status = ticket["status"]
            priority = ticket["priority"]
            topic = ticket["topic"]
            sentiment = ticket["sentiment"]

            by_status[status] = by_status.get(status, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_topic[topic] = by_topic.get(topic, 0) + 1
            by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + 1

        return {
            "total_tickets": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "by_topic": by_topic,
            "by_sentiment": by_sentiment
        }

# Global instance for app usage
_ticket_agent = TicketAgent()

def process_query_with_ticket(query: str) -> Dict:
    """Wrapper function for processing queries with ticket creation"""
    return _ticket_agent.process_query_with_ticket(query)

def create_ticket(query: str, classification: Dict, response: str,
                 escalation_info: Optional[Dict] = None) -> str:
    """Wrapper function for creating tickets"""
    return _ticket_agent.create_ticket(query, classification, response, escalation_info)

def get_ticket(ticket_id: str) -> Optional[Dict]:
    """Wrapper function for getting tickets"""
    return _ticket_agent.get_ticket(ticket_id)

def list_tickets(status: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """Wrapper function for listing tickets"""
    return _ticket_agent.list_tickets(status, limit)
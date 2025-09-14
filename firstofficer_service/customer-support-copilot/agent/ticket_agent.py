import time
import random
import string
from typing import Dict
from datetime import datetime


class TicketAgent:
    def __init__(self):
        self.tickets: Dict[str, Dict] = {}
        self.ticket_counter = 0

    def generate_ticket_id(self) -> str:
        """
        Generate a unique ticket ID using timestamp and random components.
        Format: TICK-{timestamp}-{counter}-{random_suffix}
        """
        timestamp = int(time.time() * 1000)  # milliseconds
        self.ticket_counter += 1
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        ticket_id = f"TICK-{timestamp}-{self.ticket_counter:04d}-{random_suffix}"
        return ticket_id

    def create_ticket(self, query: str, classification: Dict, response: str, escalation_info: Dict = None) -> str:
        """
        Create a new ticket with minimal associated data.
        """
        ticket_id = self.generate_ticket_id()
        ticket_data = {
            "ticket_id": ticket_id,
            "query": query,
            "classification": classification,
            "response": response,
            "escalation_info": escalation_info or {},
            "created_at": datetime.now().isoformat()
        }
        self.tickets[ticket_id] = ticket_data
        return ticket_id


# ---- Global instance for use in mquery_agent.py ----
_ticket_agent = TicketAgent()


def create_ticket(query: str, classification: Dict, response: str, escalation_info: Dict = None) -> str:
    """
    Wrapper for creating tickets directly.
    """
    return _ticket_agent.create_ticket(query, classification, response, escalation_info)

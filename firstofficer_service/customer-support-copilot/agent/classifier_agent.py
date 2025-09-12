import json
import os
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

# Prompt template for ticket classification
prompt = PromptTemplate(
    input_variables=["ticket_text"],
    template=(
        "You are a Customer Support Copilot that classifies support tickets.\n"
        "For the given ticket, return a JSON object with these fields:\n"
        " - topic: one of ['How-to', 'Product', 'Connector', 'Lineage', 'API/SDK', 'SSO', 'Glossary', 'Best practices', 'Sensitive data']\n"
        " - sentiment: one of ['Frustrated', 'Curious', 'Angry', 'Neutral']\n"
        " - priority: one of ['P0', 'P1', 'P2']\n\n"
        "Ticket: {ticket_text}\n\n"
        "Return ONLY the JSON object. If unsure, use 'Unknown' for topic, 'Neutral' for sentiment, and 'P2' for priority."
    )
)

def classify_ticket(ticket_text: str) -> dict:
    formatted = prompt.format(ticket_text=ticket_text)
    response = llm.invoke(formatted)
    
    if not response or not response.text:
        raise ValueError("LLM response is empty or invalid.")
    
    # Extract JSON from LLM output (removes markdown ```json if present)
    raw_text = response.content.strip()
    cleaned = re.sub(r"```json|```", "", raw_text, flags=re.IGNORECASE).strip()
    
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Invalid JSON returned: {raw_text}")
        parsed = {"topic": "Unknown", "sentiment": "Neutral", "priority": "P2"}
    
    # Validate parsed values
    if parsed.get("topic") not in ['How-to', 'Product', 'Connector', 'Lineage', 'API/SDK', 'SSO', 'Glossary', 'Best practices', 'Sensitive data']:
        parsed["topic"] = "Unknown"
    if parsed.get("sentiment") not in ['Frustrated', 'Curious', 'Angry', 'Neutral']:
        parsed["sentiment"] = "Neutral"
    if parsed.get("priority") not in ['P0', 'P1', 'P2']:
        parsed["priority"] = "P2"
    
    return parsed

if __name__ == "__main__":
    sample_ticket = "I tried connecting to your API but keep getting a 401 error."
    result = classify_ticket(sample_ticket)
    print("Ticket:", sample_ticket)
    print("Classification:", result)

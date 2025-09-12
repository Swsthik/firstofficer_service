import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")

# Allowed tags
TOPIC_TAGS = [
    "How-to", "Product", "Connector", "Lineage",
    "API/SDK", "SSO", "Glossary", "Best practices", "Sensitive data"
]
SENTIMENT_TAGS = ["Frustrated", "Curious", "Angry"]
PRIORITY_TAGS = ["P0", "P1", "P2"]

# Prompt template for classification
classification_prompt = PromptTemplate(
    input_variables=["ticket_text"],
    template=(
        "You are a Customer Support Copilot that classifies support tickets.\n"
        "For the given ticket, return a JSON object with these fields:\n"
        f" - topic: one of {TOPIC_TAGS}\n"
        f" - sentiment: one of {SENTIMENT_TAGS}\n"
        f" - priority: one of {PRIORITY_TAGS}\n\n"
        "Ticket: {ticket_text}\n\n"
        "Return ONLY the JSON object. If unsure, use 'Unknown' for topic, 'Neutral' for sentiment, and 'P2' for priority."
    )
)

# Updated LLM without deprecated parameter
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # latest Gemini Flash model
    api_key=api_key,
    temperature=0
)

classifier_chain = LLMChain(prompt=classification_prompt, llm=llm)

def classify_ticket(ticket_text: str) -> Dict[str, Any]:
    """Classify a support ticket into topic, sentiment, and priority."""
    # Use invoke instead of deprecated run
    raw_output = classifier_chain.invoke(ticket_text)

    # Clean and parse JSON safely
    import re
    try:
        # Extract JSON block from LLM output
        match = re.search(r'\{[\s\S]*\}', str(raw_output))
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON found in output")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Warning: Invalid JSON returned: {raw_output}")
        parsed = {"topic": "Unknown", "sentiment": "Neutral", "priority": "P2"}

    # Validate parsed values
    topic = parsed.get("topic", "Unknown")
    sentiment = parsed.get("sentiment", "Neutral")
    priority = parsed.get("priority", "P2")

    if topic not in TOPIC_TAGS:
        topic = "Unknown"
    if sentiment not in SENTIMENT_TAGS:
        sentiment = "Neutral"
    if priority not in PRIORITY_TAGS:
        priority = "P2"

    return {"topic": topic, "sentiment": sentiment, "priority": priority}


if __name__ == "__main__":
    sample_ticket = "I tried connecting to your API but keep getting a 401 error."
    result = classify_ticket(sample_ticket)
    print("Ticket:", sample_ticket)
    print("Classification:", result)

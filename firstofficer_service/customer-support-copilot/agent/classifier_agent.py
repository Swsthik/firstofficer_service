from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Load local sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = None
sentiment_model = None

def load_sentiment_model():
    global sentiment_tokenizer, sentiment_model
    if sentiment_model is None:
        try:
            sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
            sentiment_model.eval()
        except Exception as e:
            print(f"[Warning] Could not load local sentiment model: {e}")
            sentiment_tokenizer = None
            sentiment_model = None

# Expanded sentiment tags and mapping
SENTIMENT_LABELS = {
    0: "Angry",      # negative
    1: "Neutral",    # neutral
    2: "Happy"       # positive
}

# Map model output to richer set
SENTIMENT_RICH_MAP = {
    "Angry": ["Angry", "Frustrated", "Sad", "Disappointed"],
    "Neutral": ["Neutral", "Uncertain", "Confused"],
    "Happy": ["Happy", "Curious", "Excited", "Grateful"]
}

def local_sentiment_analysis(text):
    load_sentiment_model()
    if not sentiment_model or not sentiment_tokenizer:
        return "Neutral"
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    base = SENTIMENT_LABELS.get(pred, "Neutral")
    # Heuristic: pick a richer tag based on keywords
    text_lower = text.lower()
    if base == "Angry":
        if any(w in text_lower for w in ["frustrated", "annoyed", "upset", "angry", "disappointed", "sad"]):
            return "Frustrated"
        return "Angry"
    elif base == "Happy":
        if any(w in text_lower for w in ["curious", "wondering", "interested", "excited"]):
            return "Curious"
        if any(w in text_lower for w in ["thank", "grateful", "appreciate"]):
            return "Grateful"
        return "Happy"
    elif base == "Neutral":
        if any(w in text_lower for w in ["confused", "uncertain", "not sure"]):
            return "Confused"
        return "Neutral"
    return base
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
    # Use LLM for topic and priority only
    formatted = prompt.format(ticket_text=ticket_text)
    response = llm.invoke(formatted)
    if not response or not response.text:
        raise ValueError("LLM response is empty or invalid.")
    raw_text = response.content.strip()
    cleaned = re.sub(r"```json|```", "", raw_text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Invalid JSON returned: {raw_text}")
        parsed = {"topic": "Unknown", "priority": "P2"}

    # Validate topic and priority
    if parsed.get("topic") not in ['How-to', 'Product', 'Connector', 'Lineage', 'API/SDK', 'SSO', 'Glossary', 'Best practices', 'Sensitive data']:
        parsed["topic"] = "Unknown"
    if parsed.get("priority") not in ['P0', 'P1', 'P2']:
        parsed["priority"] = "P2"

    # Use local model for sentiment
    sentiment = local_sentiment_analysis(ticket_text)
    parsed["sentiment"] = sentiment
    return parsed

if __name__ == "__main__":
    sample_ticket = " your system is charging me twice."
    result = classify_ticket(sample_ticket)
    print("Ticket:", sample_ticket)
    print("Classification:", result)

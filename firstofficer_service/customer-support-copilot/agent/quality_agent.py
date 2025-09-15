# agent/quality_agent.py
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
import os, re, json
from dotenv import load_dotenv

# --- Load env and init LLM ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.0
)

class QualityAgent:
    """
    Evaluates final LLM response quality.
    Decides if response is helpful, irrelevant, or missing context.
    """

    def __init__(self):
        pass

    def evaluate(self, user_query: str, final_response: str, context_found: bool) -> Dict:
        prompt = f"""
        You are a support QA evaluator.

        Task:
        1. Judge if the user's query is relevant to the Atlan support/product domain.
        2. If the query is irrelevant (e.g., "What’s the login criteria of Google"), 
           mark quality as HIGH (1.0) and escalation = False, regardless of context.
        3. If the query is relevant but no useful context was found, 
           quality = LOW (0.4) and escalation = True.
        4. If the query is relevant and the response contains sufficient details,
           quality = GOOD (0.7–1.0) and escalation = False.

        User Query:
        {user_query}

        Final AI Response:
        {final_response}

        Context Found: {context_found}

        Respond strictly in JSON with:
        {{
          "response_quality": float,
          "should_escalate": bool,
          "reasoning": ["short explanation"]
        }}
        """

        raw = llm.invoke(prompt).content.strip()

        try:
            # Extract JSON substring safely
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw_json = match.group(0)
            else:
                raw_json = raw

            result = json.loads(raw_json)

            # Normalize reasoning → always list
            if isinstance(result.get("reasoning"), str):
                result["reasoning"] = [result["reasoning"]]

        except Exception:
            # fallback
            result = {
                "response_quality": 0.5,
                "should_escalate": not context_found,
                "reasoning": ["Fallback evaluation"]
            }

        return result

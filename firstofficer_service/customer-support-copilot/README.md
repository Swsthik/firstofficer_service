# Customer Support Copilot

A Streamlit-based app for customer support ticket classification and retrieval-augmented generation (RAG).

## Structure
- `app.py`: Main Streamlit app
- `src/`: Source code modules
- `data/`: Data and knowledge base
- `tests/`: Unit tests
- `docs/`: Documentation


## Configuration

You can control the RAG agent's retrieval and escalation logic dynamically using environment variables in your `.env` file:

- `RAG_MAX_DOCS`: Number of top documents to retrieve for each query (default: 5)
- `ESCALATION_THRESHOLD`: Escalation score threshold for routing to a human agent (default: 0.6)

Example `.env` entries:

```
RAG_MAX_DOCS=5
ESCALATION_THRESHOLD=0.6
```

You can also override these values at runtime by passing `max_docs` and `escalation_threshold` as arguments to the `RAGAgent` or its `process_query` method.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from agent.classifier_agent import classify_ticket
from agent.mquery_agent import handle_message, _agent_instance as mquery_agent
from agent.rag_agent import RAGAgent
from rag import retrieval
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

rag_agent = RAGAgent()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Backend API is running"})

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Fetch sample tickets for dashboard"""
    try:
        with open('data/sample_tickets.json', 'r') as f:
            tickets = json.load(f)
        return jsonify({"tickets": tickets})
    except FileNotFoundError:
        return jsonify({"tickets": [], "error": "Sample tickets file not found"})

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a single ticket"""
    start_time = time.time()
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    ticket_text = data['text']
    try:
        classification = classify_ticket(ticket_text)
        processing_time = int((time.time() - start_time) * 1000)  # ms

        return jsonify({
            "classification": classification,
            "processing_time": processing_time,
            "confidence": 95  # Placeholder, can be enhanced
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle conversational queries with full agent analysis"""
    start_time = time.time()
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    user_query = data['query']
    try:
        # Use RAG Agent for processing
        result = rag_agent.process_query(user_query)

        processing_time = int((time.time() - start_time) * 1000)  # ms

        return jsonify({
            "query": user_query,
            "type": result['type'],
            "response": result['content'],
            "classification": result['classification'],
            "escalation_score": result['escalation_score'],
            "factors": result['factors'],
            "reasoning": result['reasoning'],
            "sources": result['sources'],
            "processing_time": processing_time,
            "confidence": 90,  # Placeholder
            "agents_used": ["rag_agent", "classifier"],  # Updated
            "analysis_details": f"Query classified as {result['classification'].get('topic', 'Unknown')} with {result['classification'].get('sentiment', 'Neutral')} sentiment. Escalation score: {result['escalation_score']:.2f}."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rag', methods=['POST'])
def rag_endpoint():
    """Direct RAG agent processing with escalation logic"""
    start_time = time.time()
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    user_query = data['query']
    try:
        result = rag_agent.process_query(user_query)
        processing_time = int((time.time() - start_time) * 1000)
        return jsonify(result | {"processing_time": processing_time})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mquery', methods=['POST'])
def mquery_endpoint():
    """Multi-query agent for conversational responses"""
    start_time = time.time()
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    user_query = data['query']
    try:
        response = mquery_agent.generate_response(user_query)
        processing_time = int((time.time() - start_time) * 1000)
        return jsonify({
            "query": user_query,
            "response": response,
            "processing_time": processing_time,
            "agent": "mquery"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
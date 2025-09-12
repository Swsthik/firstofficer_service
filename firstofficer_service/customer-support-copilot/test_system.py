"""
Test script for the updated RAG pipeline with classifier integration
"""

import sys
import os
sys.path.append('src')

from src.classifier import classify_ticket
from src.rag_pipeline import run_rag_pipeline

def test_classifier():
    """Test the classifier functionality"""
    print("=== Testing Classifier ===")
    
    test_queries = [
        "I'm having trouble setting up SSO with SAML authentication",
        "How do I use the API to get data lineage information?",
        "I'm angry about my billing charges, this is terrible!",
        "Can you help me understand how connectors work?",
        "I need urgent help with sensitive data classification"
    ]
    
    for query in test_queries:
        result = classify_ticket(query)
        print(f"\nQuery: {query}")
        print(f"Classification: {result}")

def test_rag_pipeline():
    """Test the RAG pipeline functionality"""
    print("\n\n=== Testing RAG Pipeline ===")
    
    test_queries = [
        "How do I set up SSO?",
        "What are the API rate limits?",
        "I have a billing issue and I'm frustrated"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = run_rag_pipeline(query)
            print(f"Response Type: {response.response_type}")
            print(f"Content: {response.content[:200]}...")
            print(f"Should Escalate: {response.should_escalate}")
            print(f"Sources: {response.sources}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing the updated customer support system\n")
    
    # Test classifier
    test_classifier()
    
    # Test RAG pipeline (may fail if dependencies not installed)
    test_rag_pipeline()
    
    print("\n=== Test Summary ===")
    print("✅ Classifier: Basic functionality implemented")
    print("⚠️  RAG Pipeline: Requires dependencies (faiss, sentence-transformers, etc.)")
    print("🔧 Escalation Logic: Integrated into both components")
    print("📝 Next Steps: Install requirements.txt and set OPENAI_API_KEY")
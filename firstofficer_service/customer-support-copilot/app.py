# app.py
import streamlit as st
from agent import classifier_agent
from rag import retrieval  # your retrieval.py

# Allowed topics for RAG response
RAG_TOPICS = ["How-to", "Product", "API/SDK", "SSO", "Best practices"]

st.set_page_config(page_title="Customer Support Copilot", layout="wide")
st.title("🛠 Customer Support Copilot Demo")

# --- Sidebar: Bulk Ticket Classification ---
st.sidebar.header("Bulk Ticket Classification")
uploaded_file = st.sidebar.file_uploader("Upload sample_tickets.txt or CSV", type=["txt", "csv"])
if uploaded_file:
    tickets = uploaded_file.read().decode("utf-8").splitlines()
    st.sidebar.success(f"Loaded {len(tickets)} tickets")
    
    st.subheader("📊 Ticket Classification Dashboard")
    for i, ticket in enumerate(tickets, 1):
        result = classifier_agent.classify_ticket(ticket)
        st.markdown(f"**Ticket {i}:** {ticket}")
        st.json(result)

# --- Main: Interactive AI Agent ---
st.subheader("🤖 Interactive AI Agent")
user_query = st.text_area("Enter a new ticket or customer question here:")

if st.button("Submit Query") and user_query.strip():
    with st.spinner("Analyzing ticket..."):
        # Step 1: Classify the ticket
        classification = classifier_agent.classify_ticket(user_query)
        
        st.markdown("### 🔍 Internal Analysis")
        st.json(classification)
        
        topic = classification.get("topic", "Unknown")
        
        # Step 2: Generate RAG response if topic is in allowed list
        if topic in RAG_TOPICS:
            st.markdown("### 💬 Final Response (RAG Generated)")
            answer = retrieval.retrieve_and_answer(user_query, k=3)
            st.text(answer)
        else:
            st.markdown("### 💬 Final Response")
            st.info(f"This ticket has been classified as '{topic}' and routed to the appropriate team.")

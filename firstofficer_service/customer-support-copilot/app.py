import streamlit as st
from agent import classifier_agent
from agent import mquery_agent   # new conversational agent
from rag import retrieval        # retrieval for RAG pipeline

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

# --- Main: Interactive Conversational Agent ---
st.subheader("🤖 Interactive AI Agent")

# Keep conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Agent:** {msg['content']}")

# Input box
user_query = st.text_input("Enter your message:")

if st.button("Send") and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Agent is thinking..."):
        # Use the conversational agent logic
        response = mquery_agent.handle_message(user_query, retrieval, classifier_agent, RAG_TOPICS)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


import streamlit as st
from agent import classifier_agent
from agent import mquery_agent   # new conversational agent

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

# Initialize conversation history and logs
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# Display past conversation in main panel
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Agent:** {msg['content']}")

# Input box
user_query = st.text_input("Enter your message:")

if st.button("Send") and user_query.strip():
    # Append user query to session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Agent is thinking..."):
        # Get response and structured log
        response, log_entry = mquery_agent.handle_message(user_query, return_log=True)
    
    # Append assistant response and structured log
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.logs.append(log_entry)
    
    # Rerun to refresh conversation
    st.rerun()

# --- Sidebar: Display conversation logs ---
st.sidebar.header("Conversation Logs")
for i, log in enumerate(st.session_state.logs, 1):
    st.sidebar.markdown(f"**Turn {i}:**")
    st.sidebar.json(log)

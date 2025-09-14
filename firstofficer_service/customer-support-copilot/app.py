import streamlit as st
import pandas as pd
from agent import mquery_agent   # conversational agent

st.set_page_config(page_title="Customer Support Copilot", layout="wide")
st.title("🛠 Customer Support Copilot ")

# Initialize session state containers
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ----------------------------
# Ticket Dashboard (top)
# ----------------------------
st.subheader("📊 Ticket Dashboard")

def _normalize_classification(c):
    """Return a safe dict for classification (handle 'N/A' string cases)."""
    if not isinstance(c, dict):
        return {}
    return c

def _truncate(text: str, length: int = 120):
    if not isinstance(text, str):
        return ""
    return (text[:length] + "…") if len(text) > length else text

if st.session_state.logs:
    table_rows = []
    for log in st.session_state.logs:
        classification = _normalize_classification(log.get("Classification", {}))
        assistant_resp = log.get("Assistant Response")
        # Fallbacks
        if assistant_resp is None:
            assistant_resp = log.get("Content", "")

        table_rows.append({
            "Ticket ID": log.get("Ticket ID", "-"),
            "Topic": classification.get("topic", "-"),
            "Sentiment": classification.get("sentiment", "-"),
            "Priority": classification.get("priority", "-"),
            "Response": _truncate(assistant_resp, 120),
            "Should Escalate": log.get("Should Escalate", False),
        })

    df = pd.DataFrame(table_rows)

    # Apply color coding based on escalation
    def highlight_escalation(row):
        if row.get("Should Escalate"):
            return ['background-color: #ffcccc; color: black;'] * len(row)  # light red
        else:
            return ['background-color: #ccffcc; color: black;'] * len(row)  # light green

    styled_df = df.style.apply(highlight_escalation, axis=1)

    # Hide the helper column if you don’t want to show it in the UI
    styled_df = styled_df.hide(axis="columns", subset=["Should Escalate"])

    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("No tickets generated yet. Start a conversation to see tickets here!")

# ----------------------------
# Support Agent (below dashboard)
# ----------------------------
st.subheader("🤖 Support Agent")

# Display past conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Agent:** {msg['content']}")

# Input box (preserve input with a key)
user_query = st.text_input("Enter your message:", key="user_input")

if st.button("Send") and user_query and user_query.strip():
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Agent is thinking..."):
        # Get response + structured log from the multi-query agent
        # handle_message returns (response, log) when return_log=True
        result = mquery_agent.handle_message(user_query, return_log=True)
        # Some variations of the wrapper may return just response (older versions).
        if isinstance(result, tuple) and len(result) == 2:
            response, log_entry = result
        else:
            response = result
            log_entry = {}

    # Ensure log_entry is a dict
    if log_entry is None:
        log_entry = {}

    # Add assistant's final response into log so dashboard shows the final reply
    log_entry["Assistant Response"] = response

    # Normalize missing keys to avoid issues in dashboard rendering
    if "Classification" not in log_entry:
        log_entry["Classification"] = "N/A"
    if "Ticket ID" not in log_entry:
        # preserve existing None if present; otherwise set None
        log_entry["Ticket ID"] = log_entry.get("Ticket ID", None)

    # Save assistant response + structured log
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.logs.append(log_entry)

    # Refresh UI
    st.rerun()

# ----------------------------
# Sidebar: Raw conversation logs
# ----------------------------
st.sidebar.header("Conversation Logs (raw)")
for i, log in enumerate(st.session_state.logs, 1):
    st.sidebar.markdown(f"**Turn {i}:**")
    st.sidebar.json(log)

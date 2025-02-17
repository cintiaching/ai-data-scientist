import time
from uuid import uuid4

import streamlit as st
from agents.supervisor import get_ai_data_scientist

ai_data_scientist = get_ai_data_scientist()

st.title("🤖 AI Data Scientist Chatbot")


def stream_response(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid4()

if st.button("New Session 🧹", type="primary"):
    st.session_state.messages = []
    st.session_state.question_history = []
    st.session_state.thread_id = uuid4()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything about your data!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ai_data_scientist.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        config={"thread_id": st.session_state.thread_id}
    )
    final_response = response["messages"][-1].content

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write_stream(stream_response(final_response))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})

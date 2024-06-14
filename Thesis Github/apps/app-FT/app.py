import streamlit as st
import time
from tune import chatbot_response

st.title('Chat with your AI counselor - Chatbot 1')

# Initialize chat history, memory, and memory_and_summary
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "memory_and_summary" not in st.session_state:
    st.session_state.memory_and_summary = ''

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from chatbot and update memory variables
    response, st.session_state.memory, st.session_state.memory_and_summary = chatbot_response(prompt, st.session_state.memory, st.session_state.memory_and_summary)

    # Simulate chatbot is typing each token
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        displayed_text = ""
        for token in response.split():  # Splitting response into words (tokens)
            displayed_text += token + " "  # Add token and a space
            typing_placeholder.markdown(displayed_text)
            time.sleep(0.05)  # Short delay between tokens
        typing_placeholder.markdown(response)  # Display full message

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

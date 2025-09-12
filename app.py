import streamlit as st
from src.inference import load_inference_model, generate_response

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Chatbot 🩺",
    page_icon="🤖",
    layout="centered"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def get_model_and_tokenizer():
    return load_inference_model()

model, tokenizer = get_model_and_tokenizer()

# --- App UI ---
st.title("Qwen 2.5 Medical Chatbot 🩺")
st.markdown("Ask me any medical question. I'm here to help!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is your medical question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(model, tokenizer, prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
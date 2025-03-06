import os
import streamlit as st
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Ensure the "models" directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Define file paths (Fixed Windows Path Issue)
EMBEDDINGS_PATH = r"C:\Users\LENOVO\Downloads\archive\question_embeddings.npy"
FAISS_INDEX_PATH = r"C:\Users\LENOVO\Downloads\archive\faiss_index.bin"
DATASET_PATH = r"C:\Users\LENOVO\Downloads\archive\medquad.csv"

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    st.error(f"Dataset file not found at: {DATASET_PATH}")
    st.stop()

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if embeddings and FAISS index exist
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
    st.write("✅ Loading saved FAISS index and embeddings...")
    question_embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    st.write("⚙️ Computing embeddings and creating FAISS index...")
    
    # Convert questions to vector embeddings
    question_embeddings = np.array([embed_model.encode(q) for q in df["question"]])
    
    # Save embeddings
    np.save(EMBEDDINGS_PATH, question_embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(question_embeddings)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

st.write("✅ FAISS index ready!")

# Configure Gemini API (Replace with your actual API key)
genai.configure(api_key="AIzaSyCP_0PsR77gajEW1RP6ZV66kzCiNed8rzk")
model = genai.GenerativeModel("gemini-pro")

# Function to search FAISS and generate response
def get_answer(user_question):
    try:
        user_embedding = np.array([embed_model.encode(user_question)])
        _, closest_idx = index.search(user_embedding, 1)
        best_match = df.iloc[closest_idx[0][0]]["answer"]
        
        # Generate response using Gemini
        prompt = (f"User asked: {user_question}\n\n"
                  f"Here is a relevant answer from our dataset:\n{best_match}\n\n"
                  "Please improve and elaborate on this response.")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.title("🩺 Health Insight Chatbot")
st.write("Ask me any medical question, and I'll provide a detailed response!")

# Maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add a "Clear Chat" button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# User input
user_input = st.chat_input("Ask a medical question...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Show typing indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 *Thinking...*")
    
    # Generate AI response
    response = get_answer(user_input)
    
    # Replace typing indicator with actual response
    message_placeholder.markdown(response)
    
    # Save AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

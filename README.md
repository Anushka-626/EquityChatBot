# EquityChatBot

Desciption

# Local Chatbot with URL Data (Ollama + LangChain)

A fully local, privacy-friendly chatbot that can read and answer questions based on news articles or blog posts from the web. Powered by [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [Streamlit](https://streamlit.io/), this app requires no OpenAI API key and runs everything on your machine.

# Features

- Ask questions based on the content of 1–3 URLs
- Fast, local inference using Ollama models like `mistral`
- 100% private – no cloud model calls
- Uses LangChain + FAISS for document retrieval and QA
- Simple and interactive Streamlit interface

## 📁 Project Structure

Install Python dependencies

bash
pip install -r requirements.txt

# Install and start Ollama

Download and install Ollama from: https://ollama.com/download

Pull the Mistral model:

bash
ollama pull mistral
ollama serve

# Running the App
bash
streamlit run main.py

# How It Works
1. You input 1–2 article URLs.
2. The app fetches and splits the text into chunks.
3. FAISS + Mistral embeddings index the content locally.
4. You ask a question — the app searches relevant chunks and gets an answer from Mistral running on your machine.

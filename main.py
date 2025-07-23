import os
import time
import pickle
import streamlit as st

from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OllamaEmbeddings

# Page config
st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("Local Chatbot with URL Data")

# Sidebar for URLs
st.sidebar.header("🔗 Enter News Article URLs (1–3)")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("📄 Process URLs")

file_path = "faiss_store_ollama.pkl"

#Process URL
if process_url_clicked:
    # Clean empty URLs
    valid_urls = [u for u in urls if u.strip() != ""]
    
    if not valid_urls:
        st.error("❌ Please enter at least one valid URL.")
        st.stop()

    # Load content
    try:
        loader = UnstructuredURLLoader(urls=valid_urls)
        with st.spinner("🔄 Loading and processing data from URLs..."):
            data = loader.load()
    except Exception as e:
        st.error(f"❌ Error loading URLs: {e}")
        st.stop()

    if not data:
        st.error("⚠️ No data could be loaded from the provided URLs.")
        st.stop()

    # Split content
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("❌ No readable text found in the URLs. Try different ones.")
        st.stop()

    # Show debug info
    st.success(f"✅ {len(docs)} text chunks created from {len(valid_urls)} URLs.")

    # Generate embeddings
    embeddings = OllamaEmbeddings(model="mistral")  # You can change to 'mistral' or 'llama3'
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("🎉 URLs processed and embedded successfully!")

#Ask a Question
query = st.text_input("Ask a question based on the URLs above:")

if query:
    if os.path.exists(file_path):
        with st.spinner("🤖 Generating answer from local LLM..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=False  # False for now
            )

            try:
                result = chain.run(query)
                st.subheader("Answer")
                st.write(result)
            except Exception as e:
                st.error(f"Error during answer generation: {e}")
    else:
        st.error("⚠️ You need to process URLs before asking questions.")

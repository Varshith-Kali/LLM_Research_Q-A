import os
import streamlit as st
import pickle
import time
import langchain
import google.generativeai as palm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ["GOOGLE_API_KEY"]
palm.configure(api_key = api_key)


st.title("LLM based Research Q&A Tool :")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

button_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main = st.empty()

llm = GooglePalm()
llm.temperature = 0.4


if button_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2000
    )

    main.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = GooglePalmEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)
    main.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


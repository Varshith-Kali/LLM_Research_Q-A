import os
import streamlit as st
import pickle
import time
# import google.generativeai as palm
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv() # read local .env file
# palm.configure(api_key = api_key)


st.title("LLM based Research Q&A Tool :")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter URL {i+1}")
    urls.append(url)

button_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main = st.empty()

#llm = langchain_google_genai.GoogleGenerativeAI(google_api_key = os.environ["GOOGLE_API_KEY"], temperature = 0.2)
llm = GoogleGenerativeAI(model="gemini-pro", show_progress_bar=True, google_api_key=os.environ["GOOGLE_API_KEY"])
# print(
#     llm.invoke(
#         "What are some of the pros and cons of Python as a programming language?"
#     )
# )
# llm.temperature = 0.4


if button_clicked:
    # load data
    loader = SeleniumURLLoader(urls=urls)
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
    # embeddings = GooglePalmEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
    print(docs)
    print(embeddings)
    vector_index = FAISS.from_documents(docs, embeddings)
    # vector_index = FAISS.IndexFlatL2(len(embeddings[0]))
    main.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)


query = main.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)


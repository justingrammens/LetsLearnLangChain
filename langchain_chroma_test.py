import os, tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

from langchain.chains import RetrievalQA

# Streamlit app
st.subheader('Ask Questions of Your Documents!')

source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

query = st.text_input("Enter your query")

# If the 'Summarize' button is clicked
if st.button("Ask Question"):
    # Validate inputs
    try:
        with st.spinner('Please wait...'):
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)

            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings()
            vectordb = Chroma.from_documents(pages, embeddings)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
            
            retriever = vectordb.as_retriever()            
            chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type='stuff')
            
            summary = chain.run(query)
            st.success(summary)
    except Exception as e:
        st.exception(f"An error occurred: {e}")

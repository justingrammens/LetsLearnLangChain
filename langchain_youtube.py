import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import logging

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def main():

    #logging.basicConfig(level=logging.DEBUG)
    st.title("YouTube Transcript Chatbot")
    
    #https://www.youtube.com/watch?v=dQw4w9WgXcQ
    #Based on the document, what are you never gonna do?

    # User input fields
    youtube_url = st.text_input("Enter YouTube URL")
    question = st.text_input("Enter Your Question")
    
    if st.button("Get Transcript"):

        if youtube_url:

            # Load YouTube video transcript
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            # create documents from the transcript
            documents = loader.load()
            # split out the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
            chunks = text_splitter.split_documents(documents)

            # create embeddings from the chunks
            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(chunks, embeddings)
            
            #  create our llm and retriever
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
            retriever=vector_store.as_retriever()

            crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

            st.session_state.crc = crc
            st.success('File Uploaded and chunked successfully')
        else:
            st.warning("Please enter a YouTube URL")

        if question:
            if 'crc' in st.session_state:
                crc = st.session_state.crc

                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                response = crc.run({'question': question, 
                                   'chat_history': st.session_state['history']
                                   })
                st.session_state['history'].append((question, response))
                # write the response
                #st.write(response)

                for prompts in st.session_state['history']:
                    # then we write the question and answer
                    st.write("Question: " + prompts[0])
                    st.write("Answer: " + prompts[1])
        else:
            st.warning("Please enter a question")
        
if __name__ == "__main__":
    main()

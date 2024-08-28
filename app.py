import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Custom CSS for improved UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        caret-color: #FF4B4B;
    }
    .stChat {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This is a RAG Chatbot application built on the Gemini model. "
    "It has knowledge about the YOLOv9 paper and can answer questions related to it."
)
st.sidebar.title("How to use")
st.sidebar.write(
    "1. Type your question in the chat input box at the bottom of the page.\n"
    "2. Press Enter to submit your question.\n"
    "3. The chatbot will process your question and provide an answer based on the YOLOv9 paper."
)

# Main content
st.markdown("""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,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" alt="Logo" width="100">
        </div>
        """, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>PdfSage</h1>", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load and process the PDF document
@st.cache_resource
def load_and_process_pdf():
    with st.spinner("Loading and processing the YOLOv9 paper..."):
        loader = PyPDFLoader("yolov9_paper.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    return vectorstore

vectorstore = load_and_process_pdf()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about the YOLOv9 paper:")
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Simulate stream of response with milliseconds delay
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": query})
            for chunk in response["answer"].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
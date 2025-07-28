import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma

# Streamlit UI
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 Conversational RAG with PDF + Chat History")
st.markdown("Upload PDFs, embed them, and chat with context-rich answers.")

# Input Groq API key
api_key = st.text_input("🔑 Enter your NVIDIA API Key", type="password")

# Session & Chat setup
session_id = st.text_input("🧠 Session ID", value="default_session")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Process PDFs and create vector store
uploaded_files = st.file_uploader("📤 Upload PDF files", type="pdf", accept_multiple_files=True)

if api_key:
    llm = ChatNVIDIA(api_key = api_key, model="nvidia/llama-3.3-nemotron-super-49b-v1")
    embeddings = NVIDIAEmbeddings(api_key = api_key, model="nvidia/llama-3.2-nemoretriever-300m-embed-v1")

    if uploaded_files and "vector_ready" not in st.session_state:
        with st.spinner("⚙️ Processing PDFs and creating embeddings..."):
            try:
                all_docs = []
                for uploaded_file in uploaded_files:
                    file_path = f"./temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

                splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                chunks = splitter.split_documents(all_docs)

                vectorstore = Chroma.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever()
                st.session_state.retriever = retriever
                st.session_state.vector_ready = True
                st.success("✅ PDFs processed and embedded.")
            except Exception as e:
                st.error(f"❌ Error while embedding: {e}")

    # Set up conversational RAG if retriever is ready
    if "vector_ready" in st.session_state:
        retriever = st.session_state.retriever

        # Contextualizer
        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and latest user question, reformulate it into a standalone question. Only return the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextual_prompt)

        # QA Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for Q&A tasks. Use the retrieved context below to answer briefly.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_message_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat input
        user_question = st.text_input("💬 Ask your question:")

        if user_question:
            try:
                history = get_session_history(session_id)
                response = conversational_chain.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}}
                )
                st.success(response["answer"])
                with st.expander("🗂 Chat History"):
                    for msg in history.messages:
                        st.markdown(f"**{msg.type.capitalize()}**: {msg.content}")
            except Exception as e:
                st.error(f"❌ Failed to get response: {e}")
else:
    st.warning("Please enter a valid NVIDIA API key to begin.")
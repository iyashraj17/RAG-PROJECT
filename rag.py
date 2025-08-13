import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Main Application ---

def main():
    """
    The main function that sets up and runs the Streamlit application.
    """
    st.set_page_config(page_title="Chat with Your PDF", page_icon="📄", layout="wide")
    st.title("📄 Chat with Your PDF (Fast & High-Quality)")
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        
        # Use Google Gemini API Key
        gemini_api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
        if st.button("Set API Key"):
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    st.session_state.api_key_configured = True
                    st.success("Google Gemini API Key has been set successfully!")
                except Exception as e:
                    st.error(f"Failed to configure API key: {e}")
            else:
                st.warning("Please enter a valid API Key.")

        if st.session_state.api_key_configured:
            st.success("API Key is configured.")
        else:
            st.info("Please enter your Google Gemini API Key to begin.")

        st.header("Upload Your PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

        if pdf_file and st.session_state.api_key_configured:
            if st.button("Process Document"):
                with st.spinner("Processing your document..."):
                    try:
                        text_chunks = get_pdf_text_chunks(pdf_file)
                        if not text_chunks:
                            st.error("Could not extract text. File might be empty or corrupted.")
                            return

                        st.session_state.vector_store = create_vector_store(text_chunks)
                        if not st.session_state.vector_store:
                             st.error("Failed to create vector store.")
                             return

                        st.session_state.chat_history = []
                        st.success("Document processed! You can now ask questions.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        
        st.header("Controls")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Main Chat Interface ---
    st.header("Ask a Question")

    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.info(source.page_content)

    # Handle new user input
    user_question = st.chat_input("What would you like to know about your document?")

    if user_question:
        if st.session_state.vector_store:
            handle_user_input(user_question)
        else:
            st.warning("Please upload and process a PDF document first.")

def get_pdf_text_chunks(pdf_file):
    """Loads a PDF, extracts text, and splits it into manageable chunks."""
    try:
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        os.remove(pdf_file.name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def create_vector_store(text_chunks):
    """Creates a Chroma vector store from text chunks using a free, local embedding model."""
    try:
        # Explicitly tell the embedding model to use the CPU
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return Chroma.from_documents(documents=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def handle_user_input(user_question):
    """Processes the user's question, streams the response, and updates chat history."""
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        try:
            # 1. Retrieve relevant documents from the vector store
            retriever = st.session_state.vector_store.as_retriever()
            source_documents = retriever.invoke(user_question)

            # 2. Create a prompt for the Gemini model
            context = "\n".join([doc.page_content for doc in source_documents])
            prompt = f"""Answer the following question based only on the provided context. If the context does not contain the answer, say "I don't have enough information to answer that."

            Context:
            {context}

            Question:
            {user_question}
            """

            # 3. Call the Gemini API and stream the response
            # Using the latest recommended model name
            model = genai.GenerativeModel('gemini-1.5-flash')
            response_stream = model.generate_content(prompt, stream=True)

            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            # 4. Add the final response and sources to chat history
            assistant_message = {"role": "assistant", "content": full_response, "sources": source_documents}
            st.session_state.chat_history.append(assistant_message)
            
            # 5. Display sources in an expander
            with st.expander("View Sources"):
                for source in source_documents:
                    st.info(source.page_content)

        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")


if __name__ == '__main__':
    main()

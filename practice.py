import os
import logging
from PyPDF2 import PdfReader
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings import OpenAIEmbeddings

# ==================== Configuration ====================
# Directory to store ChromaDB's persistent data
CHROMA_DIR = "./chroma_storage"

# Initialize the Chroma client with persistent storage
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))

# Get or create a collection for storing documents
collection = chroma_client.get_or_create_collection(name="chat_data")

# Set your Groq API Key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the Groq model for chatbot responses
groq_model = ChatGroq(
    model="llama3-8b-8192",  # Specify the model to use
    temperature=0.0,  # Response consistency
    max_tokens=200,  # Limit the length of the response
    max_retries=2,  # Retry if the model fails
)

# Initialize logging for debugging purposes
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# ==================== Utility Functions ====================

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    Args:
        pdf_file: The uploaded PDF file.
    Returns:
        Extracted text as a string, or None if extraction fails.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
        logger.info(f"Extracted text from {pdf_file.name} (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def add_data_to_chroma(id, document_text):
    """
    Adds document text to the ChromaDB collection.
    Args:
        id: Unique identifier for the document.
        document_text: The text content of the document.
    """
    try:
        # Generate embeddings for the document
        embeddings = OpenAIEmbeddings()
        document_vector = embeddings.embed_documents([document_text])[0]
        logger.info(f"Document ID: {id}, Vector Length: {len(document_vector)}")

        # Add the document to ChromaDB
        collection.add(
            ids=[id],
            documents=[document_text],
            embeddings=[document_vector],
            metadatas=[{"type": "pdf", "source": id}]
        )
        logger.info(f"Document added to Chroma with ID: {id}")
    except Exception as e:
        logger.error(f"Failed to add document to Chroma: {e}")

def get_relevant_docs_from_chroma(query):
    """
    Retrieves relevant documents from ChromaDB based on a user query.
    Args:
        query: The user's search query.
    Returns:
        List of relevant documents or an empty list if none are found.
    """
    try:
        # Generate embeddings for the query
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed_documents([query])[0]
        logger.info(f"Query Vector Length: {len(query_vector)}")

        # Query ChromaDB for similar documents
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3  # Number of relevant documents to return
        )
        logger.info(f"Query Results: {results}")
        return results['documents']
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []

# ==================== Streamlit UI ====================

# Streamlit title
st.title("Chatbot with Chroma Vector DB Integration")

# Chat history to provide context
if "history" not in st.session_state:
    st.session_state["history"] = []

def log_message(message):
    """
    Logs a message to both the logger and Streamlit's session state.
    """
    logger.info(message)
    st.session_state["history"].append(message)

# Chatbot interaction
user_input = st.text_input("Ask a question:")
if user_input:
    st.session_state["history"].append(f"User: {user_input}")
    
    # Fetch relevant documents from ChromaDB
    relevant_docs = get_relevant_docs_from_chroma(user_input)
    context = "\n".join(relevant_docs)  # Combine relevant docs as context

    if context:
        # Create a prompt for the model using the retrieved context
        prompt = f"Here is some information that might help you answer the question:\n{context}\nAnswer the following question: {user_input}"
        response = groq_model.run(prompt)
        st.session_state["history"].append(f"Bot: {response}")
        st.write("Response from Bot:", response)
    else:
        st.write("No relevant documents found!")

# Display chat history
st.subheader("Chat History")
for message in st.session_state["history"]:
    st.write(message)

# Upload PDFs to add to ChromaDB
st.subheader("Upload PDFs to Add to ChromaDB")
uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            document_id = f"pdf_{uploaded_file.name}"
            add_data_to_chroma(document_id, pdf_text)
            st.success(f"Document {uploaded_file.name} uploaded and added to ChromaDB.")
        else:
            st.error(f"Failed to process {uploaded_file.name}")

# Manual Query Test
st.subheader("Test Query with ChromaDB")
test_query = st.text_input("Enter a test query to search for relevant documents:")
if test_query:
    relevant_docs_test = get_relevant_docs_from_chroma(test_query)
    if relevant_docs_test:
        st.write("Relevant documents found:")
        for doc in relevant_docs_test:
            st.write(doc)
    else:
        st.write("No relevant documents found.")
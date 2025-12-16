import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# --- CONFIGURATION ---
# We use Google's model (768 dimensions) to match your DB
EMBEDDING_MODEL = "gemini-embedding-001"
TABLE_NAME = "documents"

# 1. Initialize Clients
# Make sure GEMINI_API_KEY and SUPABASE keys are in secrets.toml
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

supabase_client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Connect LangChain to Supabase
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name=TABLE_NAME,
    query_name="match_documents"
)

# --- FUNCTIONS ---

def ingest_pdf(uploaded_file):
    """
    Reads a PDF, splits it into chunks, and saves to Supabase.
    """
    try:
        # Step A: Save temp file (PyPDF needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Step B: Load and Split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Splitter: Cuts text into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 1000 chars is good for Google
            chunk_overlap=200 
        )
        splits = text_splitter.split_documents(docs)

        # Step C: Upload to Vector DB
        vector_store.add_documents(splits)
        
        # Cleanup
        os.remove(tmp_path)
        return f"✅ Success! Added {len(splits)} chunks to memory."
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def query_knowledge_base(query: str):
    """
    Searches Supabase for relevant chunks.
    """
    try:
        results = vector_store.similarity_search(query, k=4)
        if not results:
            return None
        
        # Combine the content
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
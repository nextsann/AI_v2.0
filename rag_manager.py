import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# --- CONFIGURATION ---
# Using the model you have available (3072 dimensions)
EMBEDDING_MODEL = "models/gemini-embedding-001" 

# 1. Initialize Clients
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

# This is the raw client we will use to bypass the error
supabase_client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Keep this for ingestion (since that part works)
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# --- FUNCTIONS ---

def ingest_pdf(uploaded_file):
    """
    Ingests PDF. (This part was working fine, so we keep using LangChain here)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200 
        )
        splits = text_splitter.split_documents(docs)

        vector_store.add_documents(splits)
        
        os.remove(tmp_path)
        return f"✅ Success! Added {len(splits)} chunks to memory."
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def query_knowledge_base(query: str):
    """
    MANUAL OVERRIDE:
    We call Supabase directly to avoid the 'SyncRPCFilterRequestBuilder' error.
    """
    try:
        # 1. Convert text query to vector numbers (Using Google)
        query_vector = embeddings.embed_query(query)

        # 2. Call the Database Function directly (Bypassing LangChain wrapper)
        # This uses the raw Supabase client, which doesn't have the bug.
        response = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.0, # Zero threshold = Find anything (Good for debugging)
                "match_count": 5
            }
        ).execute()

        # 3. Extract the text
        if not response.data:
            return "No relevant information found in the database."
            
        # Combine the "content" field from the top results
        results_text = "\n\n---\n\n".join([item['content'] for item in response.data])
        return results_text

    except Exception as e:
        return f"Database Search Error: {str(e)}"
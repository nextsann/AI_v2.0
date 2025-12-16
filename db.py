import streamlit as st
from supabase import create_client
import uuid

# Initialize Supabase Client
# We use st.cache_resource so we don't reconnect every time you click a button
@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase_client()

def get_all_sessions():
    """
    Fetches a list of unique chat sessions.
    """
    try:
        # We group by session_id and take the first message to get the 'title'
        # Note: This is a simple query. For a real app, you might want a separate 'sessions' table.
        # Here we just grab distinct session_ids from the history.
        response = supabase.table("chat_history")\
            .select("session_id, content, created_at")\
            .order("created_at", desc=True)\
            .execute()
            
        # Quick logic to get unique sessions and a preview title
        seen = set()
        sessions = []
        for row in response.data:
            sid = row['session_id']
            if sid not in seen:
                sessions.append({
                    "id": sid,
                    "title": row['content'][:30] + "..." # Use first 30 chars of latest msg as title
                })
                seen.add(sid)
        return sessions
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []

def get_messages(session_id):
    """
    Loads all messages for a specific chat.
    """
    try:
        response = supabase.table("chat_history")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .execute()
        return response.data
    except Exception as e:
        return []

def save_message(session_id, role, content):
    """
    Saves a single message to the cloud.
    """
    try:
        data = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        supabase.table("chat_history").insert(data).execute()
    except Exception as e:
        print(f"Error saving message: {e}")

def delete_session(session_id):
    """
    Wipes a chat history.
    """
    try:
        supabase.table("chat_history").delete().eq("session_id", session_id).execute()
    except Exception as e:
        print(f"Error deleting session: {e}")
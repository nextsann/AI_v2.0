import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import json
from datetime import datetime
import pytz

# --- 1. WEB SEARCH TOOL ---
def get_search_tool():
    if "TAVILY_API_KEY" not in st.secrets:
        return None
    return TavilySearchResults(max_results=5)

# --- 2. CALENDAR TOOLS ---
# We define these as "LangChain Tools" using the @tool decorator

@tool
def list_upcoming_events() -> str:
    """Get the next 10 events from the user's primary calendar. Returns IDs for deletion."""
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=10, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])
        
        if not events: return "No upcoming events found."
        return "\n".join([f"ID: {e['id']} | {e['start'].get('dateTime', e['start'].get('date'))}: {e['summary']}" for e in events])
    except Exception as e:
        return f"Error listing events: {e}"

@tool
def create_calendar_event(summary: str, start_time: str, end_time: str) -> str:
    """
    Create a new calendar event. 
    Args:
        summary: Title of the event.
        start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS).
        end_time: End time in ISO format.
    """
    try:
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('calendar', 'v3', credentials=creds)
        
        event = {
            'summary': summary,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'}, # Consider fixing timezone later
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error creating event: {e}"

# Group the calendar tools
calendar_tools = [list_upcoming_events, create_calendar_event]
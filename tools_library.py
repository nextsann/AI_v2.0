import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import json
from datetime import datetime
import base64
from email.mime.text import MIMEText

# --- 1. WEB SEARCH TOOL ---
def get_search_tool():
    if "TAVILY_API_KEY" not in st.secrets:
        return None
    return TavilySearchResults(max_results=5)

# --- 2. CALENDAR TOOLS ---
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
            'start': {'dateTime': start_time, 'timeZone': 'UTC'}, 
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error creating event: {e}"

# --- 3. EMAIL TOOLS (NEW) ---

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email using Gmail.
    Args:
        to: The recipient's email address.
        subject: The subject line.
        body: The plain text body of the email.
    """
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('gmail', 'v1', credentials=creds)

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        sent_message = service.users().messages().send(
            userId='me', body={'raw': raw_message}).execute()
        return f"Email sent! Message ID: {sent_message['id']}"
    except Exception as e:
        return f"Error sending email: {e}"

@tool
def read_emails(max_results: int = 5) -> str:
    """
    Read the latest UNREAD emails from the inbox.
    Args:
        max_results: Number of emails to fetch (default 5).
    """
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('gmail', 'v1', credentials=creds)

        # q='is:unread' filters for unread emails only
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread', maxResults=max_results).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No new unread emails."

        email_summaries = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = msg_data.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
            snippet = msg_data.get('snippet', '')
            
            email_summaries.append(f"From: {sender} | Subject: {subject} | Snippet: {snippet}")

        return "\n---\n".join(email_summaries)
    except Exception as e:
        return f"Error reading emails: {e}"

# Group the tools
calendar_tools = [list_upcoming_events, create_calendar_event]
email_tools = [send_email, read_emails]
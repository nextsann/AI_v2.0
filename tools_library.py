import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import json
from datetime import datetime
import base64
from email.mime.text import MIMEText

# --- 1. WEB SEARCH TOOL ---
def get_search_tool():
    if "TAVILY_API_KEY" not in st.secrets: return None
    return TavilySearchResults(max_results=5)

# --- 2. CALENDAR TOOLS ---

# Pydantic Schema: Forces the model to provide these exact fields
class CreateEventSchema(BaseModel):
    summary: str = Field(description="The title of the event")
    start_time: str = Field(description="Start time in ISO format (YYYY-MM-DDTHH:MM:SS)")
    end_time: str = Field(description="End time in ISO format (YYYY-MM-DDTHH:MM:SS)")

@tool(args_schema=CreateEventSchema)
def create_calendar_event(summary: str, start_time: str, end_time: str) -> str:
    """Create a new calendar event in the primary calendar."""
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
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
    except Exception as e: return f"Error creating event: {e}"

@tool
def list_upcoming_events() -> str:
    """Get the next 10 events from the user's primary calendar."""
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.utcnow().isoformat() + 'Z'
        events = service.events().list(calendarId='primary', timeMin=now, maxResults=10, singleEvents=True, orderBy='startTime').execute().get('items', [])
        
        if not events: return "No upcoming events found."
        return "\n".join([f"ID: {e['id']} | {e['start'].get('dateTime', e['start'].get('date'))}: {e['summary']}" for e in events])
    except Exception as e: return f"Error listing events: {e}"

# --- 3. EMAIL TOOLS ---

# Pydantic Schema: Forces the model to ask for recipients if missing
class SendEmailSchema(BaseModel):
    to: str = Field(description="The recipient's email address")
    subject: str = Field(description="The subject line of the email")
    body: str = Field(description="The plain text body content of the email")

@tool(args_schema=SendEmailSchema)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using Gmail."""
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('gmail', 'v1', credentials=creds)

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        sent = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        return f"Email sent! ID: {sent['id']}"
    except Exception as e: return f"Error sending email: {e}"

@tool
def read_emails(max_results: int = 5) -> str:
    """Read latest UNREAD emails from the inbox."""
    try:
        if "GOOGLE_TOKEN" not in st.secrets: return "Error: No Google Token found."
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        creds = Credentials.from_authorized_user_info(token_info)
        service = build('gmail', 'v1', credentials=creds)

        results = service.users().messages().list(userId='me', q='is:unread', maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        if not messages: return "No new unread emails."
        
        summaries = []
        for msg in messages:
            data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
            summaries.append(f"From: {sender} | Subject: {subject} | Snippet: {data.get('snippet', '')}")
            
        return "\n---\n".join(summaries)
    except Exception as e: return f"Error reading emails: {e}"

calendar_tools = [list_upcoming_events, create_calendar_event]
email_tools = [send_email, read_emails]
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import os
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import pipeline
import random
import re
from google.cloud.firestore import ArrayUnion

# Initialize Firebase
cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")

try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"🔥 Firebase Initialization Failed: {e}")
    db = None # Prevent crashes by setting db to None
if db is None:
    raise HTTPException(status_code=500, detail="Database connection failed")



app = FastAPI()

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 🔹 AI Defaults
DEFAULT_AI_NAME = "Hello World"
DEFAULT_PFP = ""

# 🔹 Riddle List (Can be stored in Firestore later)
RIDDLES = [
    {"question": "The more you take, the more you leave behind. What am I?", "answer": "Footsteps"},
    {"question": "I speak without a mouth and hear without ears. What am I?", "answer": "An echo"},
]


# 🟢 1️⃣ Store Chat Messages
@app.post("/chat/{user_id}")
async def store_chat(user_id: str, message: str):
    today = datetime.now().strftime("%Y-%m-%d")
    user_ref = db.collection("users").document(user_id)

    # Ensure user exists
    user_ref.set({"created_at": firestore.SERVER_TIMESTAMP}, merge=True)

    # Add message with correct timestamp handling
    doc_ref = user_ref.collection("chats").document(today)
    doc = doc_ref.get()

    if doc.exists:
        messages = doc.to_dict().get("messages", [])
    else:
        doc_ref.set({"messages": []}) # Initialize if missing

    doc_ref.update({"messages": ArrayUnion([{"text": message, "timestamp": firestore.SERVER_TIMESTAMP}])})

    return {"message": "Chat stored successfully"}



# 🟢 2️⃣ Set AI Name & PFP (Only one version)

def sanitize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip() # Removes non-alphanumeric characters

@app.post("/profile/{user_id}")
async def set_ai_profile(user_id: str, name: str = DEFAULT_AI_NAME, pfp: str = DEFAULT_PFP):
    sanitized_name = sanitize_text(name) # Ensure sanitization of the name
    sanitized_pfp = sanitize_text(pfp) # Ensure sanitization of the profile picture URL
    doc_ref = db.collection("users").document(user_id).collection("profile").document("settings")
    doc_ref.set({"name": sanitized_name, "pfp": sanitized_pfp}, merge=True)
    return {"message": "AI profile updated"}


# 🟢 4️⃣ Summarize Last 7 Days of Chats
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") # Load once

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

@app.get("/summarize/{user_id}")
async def summarize_chats(user_id: str):
    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)

    # Fetch all messages from the last 7 days in a single query
    docs = db.collection("users").document(user_id).collection("chats") \
        .where("timestamp", ">=", seven_days_ago).stream()

    messages = []
    for doc in docs:
        doc_data = doc.to_dict()
        messages.extend([msg["text"] for msg in doc_data.get("messages", [])])

    if not docs: 
        return {"summary": "No messages found in the past 7 days."}

    text = " ".join(set(messages)) # Remove duplicate messages
    chunks = chunk_text(text) # Split long text

    # Summarize each chunk
    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]

    return {"summary": " ".join(summaries)}
# services/rag_service.py

import os
import re
import sqlite3
import requests
from dotenv import load_dotenv

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import resend
import logging

# ========================================
# 1. LOAD ENV VARIABLES
# ========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "moinsystems_knowledge")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")

# ========================================
# 2. INITIALIZE ASYNC CLIENTS
# ========================================

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

qdrant = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

resend.api_key = RESEND_API_KEY

# ========================================
# 3. NORMALIZE NAME
# ========================================

def normalize_name(name: str) -> str:
    name = name.strip()
    parts = name.split()
    normalized_parts = [p.capitalize() for p in parts]
    return " ".join(normalized_parts)
# ========================================
# 4. LEAD VALIDATION
# ========================================

def extract_email(text):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_name(text: str) -> str:
    text = text.strip()
    patterns = [
        r"^my name is\s+",
        r"^i am\s+",
        r"^i'm\s+",
        r"^this is\s+",
        r"^name is\s+"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_phone(text: str) -> str:
    pattern = r"\d[\d\s-]{6,}\d"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def validate_lead(lead: dict) -> list[str]:
    errors = []

    if not lead.get("name", "").strip():
        errors.append("Name is required.")

    raw_email_text = lead.get("email", "").strip()
    extracted_email = extract_email(raw_email_text)

    if not extracted_email:
        errors.append("Please provide a valid email address.")
    else:
        lead["email"] = extracted_email

    if not lead.get("phone", "").strip():
        errors.append("Phone number is required.")

    return errors
# ========================================
# 5. MOCK EMAIL SENDING (SAFE FOR GITHUB)
# ========================================

def send_lead_email(lead: dict):
    errors = validate_lead(lead)
    if errors:
        return False, errors

    # STEP 1: Database mein save karo
    try:
        connection = sqlite3.connect("lead.db")
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lead(
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT
            )
        """)

        cursor.execute(
            "INSERT INTO lead (name, email, phone) VALUES (?, ?, ?)",
            (lead["name"], lead["email"], lead["phone"])
        )

        connection.commit()

    except Exception as e:
        logger.error(f"Database error: {e}")
        return False, "Failed to save lead to database"

    # STEP 2: n8n ko webhook bhejo
    try:
        n8n_url = os.getenv("N8N_WEBHOOK_URL")
        response = requests.post(n8n_url, json=lead)

        if response.status_code == 200:
            return True, "Lead saved and notification sent successfully"
        else:
            return False, f"Lead saved, but webhook failed with status {response.status_code}"

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return False, "Lead saved, but failed to notify n8n"
    
# ========================================
# 6. LEAD STATE
# ========================================

lead_state = {
    "active": False,
    "name": "",
    "email": "",
    "phone": ""
}

# ========================================
# 7. ASYNC RETRIEVAL (QDRANT)
# ========================================

async def retrieve_context(question: str, top_k: int = 5):
    question_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_embedding = question_response.data[0].embedding

    search_results = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=question_embedding,
        limit=top_k,
        with_payload=True
    )

    return search_results.points

# ========================================
# 8. ASYNC RAG CHAT (GPT-5-MINI)
# ========================================

async def rag_chat(question: str, top_k: int = 3, threshold: float = 0.4) -> str:
    results = await retrieve_context(question, top_k=top_k)

    relevant_results = [
        result for result in results
        if result.score >= threshold
    ]

    if not relevant_results:
        return (
            "I don't have enough verified information in my "
            "knowledge base to answer that accurately."
        )

    context = "\n\n".join(
        result.payload["text"]
        for result in relevant_results
    )

    prompt = f"""
You are the public website AI assistant for MoinSystems AI.

Answer the user's question using ONLY the information provided
in the knowledge base context below.

Knowledge base context:
{context}

User question:
{question}
"""

    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text.strip()

# ========================================
# 9. ASYNC INTENT DETECTION
# ========================================

async def detect_lead_intent(question: str) -> bool:
    prompt = f"""
Detect buying intent.

User message:
{question}

Return YES or NO.
"""

    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    result = response.output_text.strip().upper()
    return "YES" in result

# ========================================
# 10. SUCCESS MESSAGE
# ========================================

def build_lead_success_message(lead: dict) -> str:
    name = lead.get("name", "").strip() or "there"
    return (
        f"Thank you, {name}.\n"
        "Your details have been received and verified.\n"
        "A member of our team will contact you shortly to assist you further."
    )


# ========================================
# 11. HANDLE LEAD CAPTURE (SYNC)
# ========================================

def handle_lead_capture(message: str) -> str:
    if not lead_state["name"]:
        extracted_name = extract_name(message.strip())
        lead_state["name"] = normalize_name(extracted_name)
        return "Thanks! What is your email address?"

    if not lead_state["email"]:
        extracted_email = extract_email(message.strip())
        lead_state["email"] = extracted_email if extracted_email else message.strip()

        errors = validate_lead({
            "name": lead_state["name"],
            "email": lead_state["email"],
            "phone": lead_state["phone"]
        })

        email_errors = [
            error for error in errors
            if "email" in error.lower()
        ]

        if email_errors:
            lead_state["email"] = ""
            return "That doesn't look like a valid email address. Please enter your email again."

        return "Thanks! Finally, what is your contact number."

    if not lead_state["phone"]:
        extracted_phone = extract_phone(message.strip())
        lead_state["phone"] = extracted_phone if extracted_phone else message.strip()

        lead = {
            "name": normalize_name(lead_state["name"]),
            "email": lead_state["email"],
            "phone": lead_state["phone"]
        }

        errors = validate_lead(lead)

        if errors:
            lead_state["phone"] = ""
            return "Please provide a valid contact number."

        success, response = send_lead_email(lead)

        if success:
            lead_state["active"] = False
            lead_state["name"] = ""
            lead_state["email"] = ""
            lead_state["phone"] = ""

            return build_lead_success_message(lead)

        return (
            "Your details are valid, but I wasn't able to complete "
            "the notification. Please try again later."
        )

    return "Thank you. Our team will review your details."
# ========================================
# 12. MAIN CHATBOT PIPELINE (ASYNC)
# ========================================



logger = logging.getLogger(__name__)

async def chatbot(message: str) -> str:
    message = message.strip()

    if not message:
        return "Please enter a message."

    try:
        if lead_state["active"]:
            return handle_lead_capture(message)

        if await detect_lead_intent(message):
            lead_state["active"] = True
            return (
                "I'd be happy to help you get started. "
                "Could you please provide your full name?"
            )

        return await rag_chat(message)

    except Exception as e:
        logger.error(f"Chatbot pipeline failed: {e}")
        return (
            "Sorry, I'm having trouble processing your request right now. "
            "Please try again in a moment."
        )
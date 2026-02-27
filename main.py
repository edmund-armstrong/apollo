import os
import json
import uuid
import base64
import asyncio
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import httpx
import anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

app = FastAPI(title="Apollo Travel Planner")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

STORAGE_DIR = Path("storage")
UPLOADS_DIR = STORAGE_DIR / "uploads"
INDEX_FILE = STORAGE_DIR / "index.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

GMAIL_TOKEN_FILE = STORAGE_DIR / "gmail_token.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_REDIRECT_URI = "http://localhost:8000/api/auth/gmail/callback"


def gmail_flow() -> Flow:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google credentials not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env")
    return Flow.from_client_config(
        {"web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GMAIL_REDIRECT_URI],
        }},
        scopes=GMAIL_SCOPES,
        redirect_uri=GMAIL_REDIRECT_URI,
    )


def get_gmail_service():
    if not GMAIL_TOKEN_FILE.exists():
        raise HTTPException(status_code=401, detail="Gmail not connected")
    token_data = json.loads(GMAIL_TOKEN_FILE.read_text())
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_data["token"] = creds.token
        GMAIL_TOKEN_FILE.write_text(json.dumps(token_data))
    return build("gmail", "v1", credentials=creds)


def extract_gmail_body(payload: dict) -> str:
    """Recursively extract plain text from a Gmail message payload."""
    import base64
    data = payload.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        nested = extract_gmail_body(part)
        if nested:
            return nested
    return ""


def load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {"trips": {}, "files": {}}


def save_index(index: dict):
    INDEX_FILE.write_text(json.dumps(index, indent=2))


class _TextExtractor(HTMLParser):
    """Strip HTML tags and return plain text."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        text = "".join(self._parts)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


async def fetch_url_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Apollo-Travel/1.0)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        parser = _TextExtractor()
        parser.feed(resp.text)
        return parser.get_text()


def _fetch_url_text_sync(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Apollo-Travel/1.0)"}
    with httpx.Client(follow_redirects=True, timeout=15) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        parser = _TextExtractor()
        parser.feed(resp.text)
        return parser.get_text()


def _url_to_title(url: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return (p.netloc + p.path).replace("www.", "").rstrip("/") or url
    except Exception:
        return url


def _update_item(trip_id: str, item_id: str, updates: dict):
    index = load_index()
    trip = index["trips"].get(trip_id)
    if not trip:
        return
    for i, item in enumerate(trip["items"]):
        if item["id"] == item_id:
            trip["items"][i].update(updates)
            save_index(index)
            return


_LINK_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "status":        {"type": "string"},
        "title":         {"type": "string"},
        "category":      {"type": "string"},
        "summary":       {"type": "string"},
        "address":       {"type": "string"},
        "city":          {"type": "string"},
        "country":       {"type": "string"},
        "price":         {"type": "string"},
        "check_in":      {"type": "string"},
        "check_out":     {"type": "string"},
        "hours":         {"type": "string"},
        "phone":         {"type": "string"},
        "cuisine":       {"type": "string"},
        "airline":       {"type": "string"},
        "flight_number": {"type": "string"},
        "departure":     {"type": "string"},
        "arrival":       {"type": "string"},
        "rating":        {"type": "string"},
        "notes":         {"type": "string"},
    },
    "required": ["status", "title", "category", "summary"],
    "additionalProperties": False,
}

_LINK_SYSTEM = (
    "You extract travel-relevant details from webpages for a travel planning app. "
    "Only include fields with clear, explicitly stated values — never guess. "
    "status: 'extracted' = found useful info; 'low_confidence' = very little info; "
    "'inaccessible' = paywall or login required. "
    "category: one of hotel | restaurant | flight | attraction | transport | other."
)


def _extract_link_sync(trip_id: str, item_id: str, url: str):
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)

        try:
            page_text = _fetch_url_text_sync(url)
        except Exception:
            page_text = None

        if not page_text or len(page_text.strip()) < 80:
            _update_item(trip_id, item_id, {
                "extraction": {
                    "status": "inaccessible",
                    "title": _url_to_title(url),
                    "category": "other",
                    "summary": "",
                },
                "content": f"Source: {url}\n[Page inaccessible — raw URL preserved]",
            })
            return

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=_LINK_SYSTEM,
            messages=[{"role": "user", "content":
                f"Extract travel details from this page.\nURL: {url}\n\nContent:\n{page_text[:6000]}"}],
            output_config={"format": {"type": "json_schema", "schema": _LINK_EXTRACTION_SCHEMA}},
        )

        ext = json.loads(response.content[0].text)

        # Build formatted content for AI context in chat / itinerary
        lines = [
            f"Source: {url}",
            f"Title: {ext['title']}",
            f"Type: {ext['category']}",
            f"Summary: {ext['summary']}",
        ]
        for k in ["address", "city", "country", "price", "check_in", "check_out",
                  "hours", "phone", "cuisine", "airline", "flight_number",
                  "departure", "arrival", "rating", "notes"]:
            if ext.get(k):
                lines.append(f"{k.replace('_', ' ').title()}: {ext[k]}")
        if ext.get("status") == "inaccessible":
            lines.append("[Note: page is behind a paywall or login — consider pasting content manually]")
        elif ext.get("status") == "low_confidence":
            lines.append("[Note: limited info extracted — consider pasting page content manually]")

        _update_item(trip_id, item_id, {
            "extraction": ext,
            "content": "\n".join(lines),
        })

    except Exception:
        _update_item(trip_id, item_id, {
            "extraction": {
                "status": "failed",
                "title": _url_to_title(url),
                "category": "other",
                "summary": "",
            },
            "content": f"Source: {url}\n[Extraction failed — raw URL preserved]",
        })


async def extract_link_background(trip_id: str, item_id: str, url: str):
    await asyncio.to_thread(_extract_link_sync, trip_id, item_id, url)


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path("static") / "index.html").read_text()


@app.get("/api/trips")
async def list_trips():
    index = load_index()
    return {"trips": list(index["trips"].values())}


@app.post("/api/trips")
async def create_trip(name: Optional[str] = Form(None)):
    index = load_index()
    trip_id = str(uuid.uuid4())[:8]
    auto_name = name.strip() if name and name.strip() else f"Trip — {datetime.now().strftime('%b %d')}"
    index["trips"][trip_id] = {
        "id": trip_id,
        "name": auto_name,
        "items": [],
        "itinerary": None,
        "metadata": {},
    }
    save_index(index)
    return index["trips"][trip_id]


@app.patch("/api/trips/{trip_id}/rename")
async def rename_trip(trip_id: str, name: str = Form(...)):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")
    index["trips"][trip_id]["name"] = name.strip()
    save_index(index)
    return {"ok": True}


@app.delete("/api/trips/{trip_id}")
async def delete_trip(trip_id: str):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")
    del index["trips"][trip_id]
    save_index(index)
    return {"ok": True}


def _extract_metadata_sync(trip_id: str):
    """Silently parse destination/dates from captured content. Runs in thread pool."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)
        index = load_index()
        trip = index["trips"].get(trip_id)
        if not trip or not trip["items"]:
            return
        text_items = [i["content"] for i in trip["items"] if i.get("content")]
        if not text_items:
            return
        combined = "\n\n".join(text_items[:5])
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": (
                "Extract from this travel content. Return ONLY valid JSON with: "
                "destination (string, city/country or empty), dates (string, range or empty), "
                "confirmed (boolean). Content:\n" + combined[:3000]
            )}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "dates": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                "required": ["destination", "dates", "confirmed"],
                "additionalProperties": False,
            }}},
        )
        metadata = json.loads(response.content[0].text)
        index = load_index()
        if trip_id not in index["trips"]:
            return
        index["trips"][trip_id]["metadata"] = metadata
        # Auto-update name if still a default and we found a destination
        current_name = index["trips"][trip_id]["name"]
        dest = metadata.get("destination", "")
        if current_name.startswith("Trip — ") and dest:
            dates = metadata.get("dates", "")
            index["trips"][trip_id]["name"] = f"{dest}{' · ' + dates if dates else ''}"
        save_index(index)
    except Exception:
        pass  # Never surface errors during capture


async def extract_metadata_background(trip_id: str):
    await asyncio.to_thread(_extract_metadata_sync, trip_id)


@app.post("/api/trips/{trip_id}/upload")
async def upload_content(
    background_tasks: BackgroundTasks,
    trip_id: str,
    content_type: str = Form(...),  # "text" | "file" | "email" | "link"
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    label: Optional[str] = Form(None),
):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    client = get_client()
    item = {"id": str(uuid.uuid4())[:8], "type": content_type, "label": label or ""}

    if content_type == "text" or content_type == "email":
        item["content"] = text
        item["file_id"] = None

    elif content_type == "link":
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        item["url"] = url
        item["content"] = f"Source: {url}"
        item["extraction"] = {"status": "pending", "title": _url_to_title(url), "category": "other", "summary": ""}
        item["file_id"] = None

    elif content_type in ("image", "file"):
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        raw = await file.read()
        mime = file.content_type or "application/octet-stream"
        uploaded = client.beta.files.upload(file=(file.filename, raw, mime))
        item["type"] = "file"
        item["file_kind"] = "document" if mime == "application/pdf" else "image"
        item["file_id"] = uploaded.id
        item["filename"] = file.filename
        item["mime"] = mime
        item["content"] = None

    index["trips"][trip_id]["items"].append(item)
    index["trips"][trip_id]["itinerary"] = None
    save_index(index)

    # Background: silently parse metadata — never blocks the response
    background_tasks.add_task(extract_metadata_background, trip_id)
    if content_type == "link":
        background_tasks.add_task(extract_link_background, trip_id, item["id"], url)

    return {"ok": True, "item": item}


@app.post("/api/trips/{trip_id}/organize")
async def organize_trip(trip_id: str):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    trip = index["trips"][trip_id]
    if not trip["items"]:
        raise HTTPException(status_code=400, detail="No content to organize")

    client = get_client()

    # Build message content from stored items
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert travel planner. I'll give you raw travel notes, "
                f"screenshots, and email snippets for a trip called '{trip['name']}'. "
                f"Please organize everything into a clear, structured itinerary with:\n"
                f"- Day-by-day breakdown (if dates are present)\n"
                f"- Key details: accommodation, transport, activities, restaurants\n"
                f"- Important info: addresses, booking references, hours, tips\n"
                f"- A 'Quick Reference' section at the top with must-know info\n\n"
                f"Here is all the raw content:"
            ),
        }
    ]

    for item in trip["items"]:
        label = f"[{item['label'] or item['type']}]"
        if item.get("file_id"):
            file_kind = item.get("file_kind", "image")
            if file_kind == "document":
                content.append({"type": "text", "text": f"\n{label} (PDF document):"})
                content.append({"type": "document", "source": {"type": "file", "file_id": item["file_id"]}})
            else:
                content.append({"type": "text", "text": f"\n{label} (screenshot/image):"})
                content.append({"type": "image", "source": {"type": "file", "file_id": item["file_id"]}})
        elif item.get("content"):
            content.append({"type": "text", "text": f"\n{label}:\n{item['content']}"})

    response = client.beta.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": content}],
        betas=["files-api-2025-04-14"],
    )

    itinerary_text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )
    index["trips"][trip_id]["itinerary"] = itinerary_text
    save_index(index)
    return {"itinerary": itinerary_text}


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    trip_id: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    history: str = Form(default="[]"),
):
    index = load_index()
    client = get_client()

    # Build location context
    location_ctx = ""
    if latitude is not None and longitude is not None:
        location_ctx = (
            f"\nUser's current GPS location: {latitude:.5f}°N, {longitude:.5f}°E. "
            f"Use this to give directions, distances, and location-aware answers."
        )

    # Build trip context
    trip_ctx = ""
    if trip_id and trip_id in index["trips"]:
        trip = index["trips"][trip_id]
        trip_ctx = f"\n\nActive trip: '{trip['name']}'"
        if trip.get("itinerary"):
            trip_ctx += f"\n\nOrganized itinerary:\n{trip['itinerary']}"
        else:
            # Fall back to raw items
            for item in trip["items"]:
                if item.get("content"):
                    trip_ctx += f"\n\n[{item['label'] or item['type']}]:\n{item['content']}"

    system = (
        "You are Apollo, an AI travel assistant. You have access to the user's saved "
        "travel plans and their real-time location. Give concise, actionable answers. "
        "When location is provided, use it to give directions or distances. "
        "Format responses clearly — use bullet points and bold for key info."
        + location_ctx
        + trip_ctx
    )

    # Parse conversation history
    prev_messages = json.loads(history)
    messages = prev_messages + [{"role": "user", "content": message}]

    async def stream_response():
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Gmail routes
# ---------------------------------------------------------------------------

@app.get("/api/auth/gmail")
async def gmail_auth():
    flow = gmail_flow()
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    return RedirectResponse(auth_url)


@app.get("/api/auth/gmail/callback")
async def gmail_callback(code: str):
    flow = gmail_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    GMAIL_TOKEN_FILE.write_text(json.dumps({
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else GMAIL_SCOPES,
    }))
    return RedirectResponse("/?gmail=connected")


@app.get("/api/auth/gmail/status")
async def gmail_status():
    return {"connected": GMAIL_TOKEN_FILE.exists()}


@app.post("/api/email/scan")
async def scan_emails(trip_id: str = Form(...), keywords: str = Form(...)):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    service = get_gmail_service()

    # Build Gmail search query from comma-separated keywords
    terms = [k.strip() for k in keywords.split(",") if k.strip()]
    query = " OR ".join(terms)

    results = service.users().messages().list(
        userId="me", q=query, maxResults=25
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        return {"emails": []}

    emails = []
    for msg in messages[:25]:
        msg_data = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()
        headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}
        emails.append({
            "id": msg["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "sender": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg_data.get("snippet", ""),
            "body": extract_gmail_body(msg_data["payload"])[:3000],
        })

    return {"emails": emails}


@app.post("/api/email/add")
async def add_email_to_trip(
    trip_id: str = Form(...),
    subject: str = Form(...),
    sender: str = Form(...),
    date: str = Form(...),
    body: str = Form(...),
):
    index = load_index()
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    item = {
        "id": str(uuid.uuid4())[:8],
        "type": "email",
        "label": subject,
        "content": f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}",
        "file_id": None,
    }
    index["trips"][trip_id]["items"].append(item)
    index["trips"][trip_id]["itinerary"] = None
    save_index(index)
    return {"ok": True, "item": item}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

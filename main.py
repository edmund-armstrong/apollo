import os
import json
import uuid
import base64
import asyncio
from pathlib import Path
from typing import Optional

import anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

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


def load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {"trips": {}, "files": {}}


def save_index(index: dict):
    INDEX_FILE.write_text(json.dumps(index, indent=2))


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
async def create_trip(name: str = Form(...)):
    index = load_index()
    trip_id = str(uuid.uuid4())[:8]
    index["trips"][trip_id] = {
        "id": trip_id,
        "name": name,
        "items": [],
        "itinerary": None,
    }
    save_index(index)
    return index["trips"][trip_id]


@app.post("/api/trips/{trip_id}/upload")
async def upload_content(
    trip_id: str,
    content_type: str = Form(...),  # "text" | "image" | "email"
    text: Optional[str] = Form(None),
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

    elif content_type == "image":
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        raw = await file.read()
        mime = file.content_type or "image/jpeg"

        # Upload to Claude Files API so we don't re-upload later
        uploaded = client.beta.files.upload(
            file=(file.filename, raw, mime),
        )
        item["file_id"] = uploaded.id
        item["filename"] = file.filename
        item["mime"] = mime
        item["content"] = None

    index["trips"][trip_id]["items"].append(item)
    # Clear cached itinerary when new content is added
    index["trips"][trip_id]["itinerary"] = None
    save_index(index)
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
        if item["type"] == "image" and item.get("file_id"):
            content.append({"type": "text", "text": f"\n{label} (screenshot/image):"})
            content.append({
                "type": "image",
                "source": {"type": "file", "file_id": item["file_id"]},
            })
        elif item.get("content"):
            content.append({
                "type": "text",
                "text": f"\n{label}:\n{item['content']}",
            })

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
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

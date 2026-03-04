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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
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
_APP_URL = os.getenv("APP_URL", "http://localhost:8000").rstrip("/")
GMAIL_REDIRECT_URI = f"{_APP_URL}/api/auth/gmail/callback"


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
        raise HTTPException(status_code=401, detail="Gmail not connected — reconnect via the Gmail tab")
    token_data = json.loads(GMAIL_TOKEN_FILE.read_text())
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    if creds.expired:
        if not creds.refresh_token:
            GMAIL_TOKEN_FILE.unlink(missing_ok=True)
            raise HTTPException(status_code=401, detail="Gmail session expired — please reconnect")
        try:
            creds.refresh(GoogleAuthRequest())
            token_data["token"] = creds.token
            GMAIL_TOKEN_FILE.write_text(json.dumps(token_data))
        except Exception as e:
            GMAIL_TOKEN_FILE.unlink(missing_ok=True)
            raise HTTPException(status_code=401, detail=f"Gmail re-auth failed — please reconnect ({e})")
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


# ---------------------------------------------------------------------------
# Per-user storage (invite-link auth)
# ---------------------------------------------------------------------------

USERS_DIR = STORAGE_DIR / "users"
USERS_DIR.mkdir(parents=True, exist_ok=True)


def user_index_file(token: str) -> Path:
    d = USERS_DIR / token
    d.mkdir(parents=True, exist_ok=True)
    return d / "index.json"


def load_user_index(token: str) -> dict:
    f = user_index_file(token)
    if f.exists():
        return json.loads(f.read_text())
    return {"trips": {}}


def save_user_index(token: str, index: dict):
    user_index_file(token).write_text(json.dumps(index, indent=2))


def get_user_token(request: Request) -> str:
    token = request.cookies.get("apollo_user")
    if not token:
        raise HTTPException(status_code=401, detail="No invite token")
    return token


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


def _is_tiktok_url(url: str) -> bool:
    return bool(re.search(r"tiktok\.com", url, re.I))


def _fetch_tiktok_oembed(url: str) -> dict:
    """Fetch TikTok video metadata via oEmbed (no API key required)."""
    with httpx.Client(follow_redirects=True, timeout=15) as client:
        resp = client.get(
            "https://www.tiktok.com/oembed",
            params={"url": url},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        return resp.json()


def _fetch_reddit_text_sync(url: str) -> str:
    """Fetch Reddit post content via the public JSON API (no auth required)."""
    import re as _re
    # Normalize to the JSON endpoint: strip query/fragment, ensure trailing slash before .json
    clean = _re.sub(r"[?#].*$", "", url).rstrip("/")
    json_url = clean + ".json"
    with httpx.Client(follow_redirects=True, timeout=15) as client:
        resp = client.get(
            json_url,
            headers={"User-Agent": "Apollo travel app (personal use)"},
        )
        resp.raise_for_status()
        data = resp.json()
    # Reddit JSON structure: [post_listing, comments_listing]
    post = data[0]["data"]["children"][0]["data"]
    title = post.get("title", "")
    selftext = post.get("selftext", "")
    subreddit = post.get("subreddit_name_prefixed", "")
    # Pull top-level comments for extra context
    comments = []
    for child in data[1]["data"]["children"][:10]:
        body = child["data"].get("body", "")
        if body and body != "[deleted]":
            comments.append(body)
    parts = [f"Title: {title}", f"Subreddit: {subreddit}"]
    if selftext:
        parts.append(f"Post: {selftext}")
    if comments:
        parts.append("Top comments:\n" + "\n\n".join(comments))
    return "\n\n".join(parts)


_FETCH_UA_LIST = [
    # Many CDNs (Akamai, Cloudflare) explicitly allow known search crawlers
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    # Standard browser UA as second attempt
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    # Bare curl as last resort
    "curl/7.88.1",
]


def _fetch_url_text_sync(url: str) -> str:
    last_exc: Exception = RuntimeError("No UA succeeded")
    for ua in _FETCH_UA_LIST:
        try:
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            with httpx.Client(follow_redirects=True, timeout=15) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                parser = _TextExtractor()
                parser.feed(resp.text)
                text = parser.get_text()
                if len(text.strip()) >= 80:
                    return text
                # Content too sparse — try next UA before giving up
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _url_to_title(url: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return (p.netloc + p.path).replace("www.", "").rstrip("/") or url
    except Exception:
        return url


def _update_item(user_token: str, trip_id: str, item_id: str, updates: dict):
    index = load_user_index(user_token)
    trip = index["trips"].get(trip_id)
    if not trip:
        return
    for i, item in enumerate(trip["items"]):
        if item["id"] == item_id:
            trip["items"][i].update(updates)
            save_user_index(user_token, index)
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
    # output_config structured outputs require ALL properties in "required"
    "required": [
        "status", "title", "category", "summary",
        "address", "city", "country", "price",
        "check_in", "check_out", "hours", "phone",
        "cuisine", "airline", "flight_number",
        "departure", "arrival", "rating", "notes",
    ],
    "additionalProperties": False,
}

_LINK_SYSTEM = (
    "You extract travel-relevant details from webpages for a travel planning app. "
    "Only include fields with clear, explicitly stated values — never guess. "
    "status: 'extracted' = found useful info; 'low_confidence' = very little info; "
    "'inaccessible' = paywall or login required. "
    "category: one of hotel | restaurant | flight | attraction | transport | other."
)


def _is_google_maps_url(url: str) -> bool:
    return bool(re.search(r"(maps\.google\.|google\.[a-z]+/maps|maps\.app\.goo\.gl|goo\.gl/maps)", url))


def _fetch_google_maps_place(url: str) -> dict:
    """
    Extract place details from a Google Maps URL.
    Resolves shortened URLs, parses the place name, and optionally enriches
    with the Places API if GOOGLE_PLACES_API_KEY is set.
    """
    # Resolve shortened / redirected URLs (goo.gl, maps.app.goo.gl, etc.)
    resolved = url
    if "goo.gl" in url:
        try:
            with httpx.Client(follow_redirects=True, timeout=10) as c:
                r = c.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resolved = str(r.url)
        except Exception:
            pass

    # Extract place name from URL path  /maps/place/PLACE_NAME/
    place_query: Optional[str] = None
    m = re.search(r"/maps/place/([^/@?]+)", resolved)
    if m:
        import urllib.parse
        place_query = urllib.parse.unquote_plus(m.group(1))
    if not place_query:
        m = re.search(r"[?&]q=([^&]+)", resolved)
        if m:
            import urllib.parse
            place_query = urllib.parse.unquote_plus(m.group(1))

    api_key = os.getenv("GOOGLE_PLACES_API_KEY")

    # No API key — return minimal extraction from the URL itself
    if not api_key:
        title = place_query or "Google Maps place"
        return {
            "status": "extracted",
            "title": title,
            "category": "attraction",
            "summary": "",
        }

    # Call Places API — Find Place
    try:
        query = place_query or resolved
        with httpx.Client(timeout=10) as c:
            r = c.get(
                "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                params={
                    "input": query,
                    "inputtype": "textquery",
                    "fields": "name,formatted_address,rating,opening_hours,formatted_phone_number,website,types,price_level,editorial_summary",
                    "key": api_key,
                },
            )
        data = r.json()
    except Exception:
        title = place_query or "Google Maps place"
        return {"status": "extracted", "title": title, "category": "attraction", "summary": ""}

    if data.get("status") != "OK" or not data.get("candidates"):
        title = place_query or "Google Maps place"
        return {"status": "extracted", "title": title, "category": "attraction", "summary": ""}

    place = data["candidates"][0]

    # Map Google types → Apollo category
    types = place.get("types", [])
    category = "attraction"
    if any(t in types for t in ["restaurant", "food", "cafe", "bar", "bakery", "meal_takeaway"]):
        category = "restaurant"
    elif any(t in types for t in ["lodging", "hotel"]):
        category = "hotel"
    elif any(t in types for t in ["airport", "transit_station", "bus_station", "train_station"]):
        category = "transport"

    hours = None
    weekday_text = place.get("opening_hours", {}).get("weekday_text")
    if weekday_text:
        hours = " | ".join(weekday_text)

    price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    price = price_map.get(place.get("price_level"))

    rating = place.get("rating")

    return {
        "status": "extracted",
        "title": place.get("name") or place_query or "Google Maps place",
        "category": category,
        "summary": (place.get("editorial_summary") or {}).get("overview", ""),
        "address": place.get("formatted_address") or "",
        "rating": str(rating) if rating else None,
        "phone": place.get("formatted_phone_number"),
        "hours": hours,
        "price": price,
    }


def _extract_link_sync(user_token: str, trip_id: str, item_id: str, url: str):
    try:
        # TikTok: use oEmbed instead of scraping
        if _is_tiktok_url(url):
            try:
                oembed = _fetch_tiktok_oembed(url)
                ext = {
                    "status": "extracted",
                    "title": oembed.get("title", "TikTok video"),
                    "category": "other",
                    "summary": f"By @{oembed.get('author_name', 'unknown')} on TikTok",
                    "thumbnail_url": oembed.get("thumbnail_url", ""),
                    "embed_html": oembed.get("html", ""),
                }
                lines = [
                    f"Source: {url}",
                    f"Title: {ext['title']}",
                    f"Creator: @{oembed.get('author_name', '')}",
                    f"Platform: TikTok",
                ]
                _update_item(user_token, trip_id, item_id, {
                    "extraction": ext,
                    "content": "\n".join(lines),
                })
            except Exception:
                _update_item(user_token, trip_id, item_id, {
                    "extraction": {"status": "failed", "title": "TikTok video", "category": "other", "summary": ""},
                    "content": f"Source: {url}\n[TikTok oEmbed failed — raw URL preserved]",
                })
            return

        # Google Maps: use Places API instead of scraping
        if _is_google_maps_url(url):
            try:
                ext = _fetch_google_maps_place(url)
                lines = [f"Source: {url}", f"Title: {ext['title']}", f"Type: {ext['category']}"]
                if ext.get("summary"):
                    lines.append(f"Summary: {ext['summary']}")
                for k in ["address", "rating", "phone", "hours", "price"]:
                    if ext.get(k):
                        lines.append(f"{k.title()}: {ext[k]}")
                _update_item(user_token, trip_id, item_id, {
                    "extraction": ext,
                    "content": "\n".join(lines),
                })
            except Exception:
                _update_item(user_token, trip_id, item_id, {
                    "extraction": {"status": "failed", "title": "Google Maps place", "category": "attraction", "summary": ""},
                    "content": f"Source: {url}\n[Google Maps extraction failed — raw URL preserved]",
                })
            return

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)

        try:
            if "reddit.com/" in url:
                page_text = _fetch_reddit_text_sync(url)
            else:
                page_text = _fetch_url_text_sync(url)
        except Exception:
            page_text = None

        if not page_text or len(page_text.strip()) < 80:
            _update_item(user_token, trip_id, item_id, {
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

        _update_item(user_token, trip_id, item_id, {
            "extraction": ext,
            "content": "\n".join(lines),
        })

    except Exception:
        _update_item(user_token, trip_id, item_id, {
            "extraction": {
                "status": "failed",
                "title": _url_to_title(url),
                "category": "other",
                "summary": "",
            },
            "content": f"Source: {url}\n[Extraction failed — raw URL preserved]",
        })


async def extract_link_background(user_token: str, trip_id: str, item_id: str, url: str):
    await asyncio.to_thread(_extract_link_sync, user_token, trip_id, item_id, url)


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_JOIN_PASSWORD = os.getenv("JOIN_PASSWORD", "ilovetravel")

_GATE_HTML = """<!DOCTYPE html>
<html><head><title>Apollo — Early Access</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
       min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
  .card{background:rgba(255,255,255,.07);backdrop-filter:blur(20px);
        border:1px solid rgba(255,255,255,.12);border-radius:24px;
        padding:48px 40px;width:100%;max-width:380px;text-align:center}
  h1{color:#fff;font-size:2rem;margin-bottom:6px;letter-spacing:-0.5px}
  .tagline{color:rgba(255,255,255,.5);font-size:0.9rem;margin-bottom:36px}
  label{display:block;text-align:left;color:rgba(255,255,255,.6);
        font-size:0.78rem;font-weight:500;text-transform:uppercase;
        letter-spacing:.06em;margin-bottom:8px}
  input{width:100%;padding:14px 16px;border-radius:12px;border:1px solid rgba(255,255,255,.15);
        background:rgba(255,255,255,.08);color:#fff;font-size:1rem;outline:none;
        transition:border-color .2s}
  input::placeholder{color:rgba(255,255,255,.3)}
  input:focus{border-color:rgba(255,255,255,.4)}
  button{margin-top:16px;width:100%;padding:14px;border-radius:12px;border:none;
         background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
         font-size:1rem;font-weight:600;cursor:pointer;transition:opacity .2s}
  button:hover{opacity:.9}
  .error{color:#ff6b6b;font-size:0.85rem;margin-top:12px;display:none}
</style></head>
<body>
<div class="card">
  <h1>✈️ Apollo</h1>
  <p class="tagline">AI-powered travel planning</p>
  <form id="form">
    <label for="pw">Early access password</label>
    <input type="password" id="pw" placeholder="Enter password" autocomplete="current-password" />
    <button type="submit">Continue</button>
    <div class="error" id="err">Incorrect password — try again.</div>
  </form>
</div>
<script>
document.getElementById('form').onsubmit = async e => {
  e.preventDefault();
  const pw = document.getElementById('pw').value;
  const res = await fetch('/api/auth/join', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'password=' + encodeURIComponent(pw)
  });
  if (res.ok) { window.location.href = '/'; }
  else { document.getElementById('err').style.display = 'block'; }
};
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if not request.cookies.get("apollo_user"):
        return HTMLResponse(_GATE_HTML)
    return (Path("static") / "index.html").read_text()


@app.post("/api/auth/join")
async def join(password: str = Form(...)):
    if password != _JOIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password")
    token = str(uuid.uuid4()).replace("-", "")
    save_user_index(token, {"trips": {}})
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("apollo_user", token, max_age=60*60*24*365, httponly=True, samesite="lax")
    return response


@app.get("/api/trips")
async def list_trips(request: Request):
    token = get_user_token(request)
    index = load_user_index(token)
    return {"trips": list(index["trips"].values())}


async def normalize_trip_name(location: str, start_date: str = "", end_date: str = "", purpose: str = "") -> str:
    try:
        client = get_client()
        prompt = f"""Given the following trip details, return a normalized trip name.

Format: {{City}} – {{Month}} {{Start Day}} to {{End Day}}, {{3-Word Purpose}}

Rules:
- City: primary city name only, no state or country
- Dates: "April 11 to 15" if same month; "March 29 to April 2" if spans months
- Purpose: condense to 3 words max using title case. If empty, omit the comma and purpose.
- Return ONLY the trip name string. No explanation, no punctuation beyond the format.

Destination: {location}
Start date: {start_date}
End date: {end_date}
Purpose: {purpose}"""
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return location or f"Trip — {datetime.now().strftime('%b %d')}"


@app.post("/api/trips/suggest-name")
async def suggest_trip_name(
    request: Request,
    location: Optional[str] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    purpose: Optional[str] = Form(None),
):
    get_user_token(request)
    name = await normalize_trip_name(
        location=(location or "").strip(),
        start_date=(start_date or "").strip(),
        end_date=(end_date or "").strip(),
        purpose=(purpose or "").strip(),
    )
    return {"name": name}


@app.post("/api/trips")
async def create_trip(
    request: Request,
    name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
    trip_id = str(uuid.uuid4())[:8]
    if name and name.strip():
        auto_name = name.strip()
    elif location and location.strip():
        auto_name = await normalize_trip_name(
            location.strip(), (start_date or "").strip(), (end_date or "").strip(), (description or "").strip()
        )
    else:
        auto_name = f"Trip — {datetime.now().strftime('%b %d')}"
    index["trips"][trip_id] = {
        "id": trip_id,
        "name": auto_name,
        "location": (location or "").strip(),
        "start_date": (start_date or "").strip(),
        "end_date": (end_date or "").strip(),
        "description": (description or "").strip(),
        "items": [],
        "itinerary": None,
        "metadata": {},
        "created_at": datetime.now().isoformat(),
    }
    save_user_index(token, index)
    return index["trips"][trip_id]


@app.patch("/api/trips/{trip_id}")
async def edit_trip(
    request: Request,
    trip_id: str,
    name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")
    trip = index["trips"][trip_id]
    if name is not None: trip["name"] = name.strip()
    if location is not None: trip["location"] = location.strip()
    if start_date is not None: trip["start_date"] = start_date.strip()
    if end_date is not None: trip["end_date"] = end_date.strip()
    if description is not None: trip["description"] = description.strip()
    save_user_index(token, index)
    return {"ok": True, "trip": trip}


@app.delete("/api/trips/{trip_id}")
async def delete_trip(request: Request, trip_id: str):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")
    del index["trips"][trip_id]
    save_user_index(token, index)
    return {"ok": True}


@app.delete("/api/trips/{trip_id}/items/{item_id}")
async def delete_item(request: Request, trip_id: str, item_id: str):
    token = get_user_token(request)
    index = load_user_index(token)
    trip = index["trips"].get(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    before = len(trip["items"])
    trip["items"] = [it for it in trip["items"] if it["id"] != item_id]
    if len(trip["items"]) == before:
        raise HTTPException(status_code=404, detail="Item not found")
    save_user_index(token, index)
    return {"ok": True}


def _extract_metadata_sync(user_token: str, trip_id: str):
    """Silently parse destination/dates from captured content. Runs in thread pool."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)
        index = load_user_index(user_token)
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
        index = load_user_index(user_token)
        if trip_id not in index["trips"]:
            return
        index["trips"][trip_id]["metadata"] = metadata
        current_name = index["trips"][trip_id]["name"]
        dest = metadata.get("destination", "")
        if current_name.startswith("Trip — ") and dest:
            dates = metadata.get("dates", "")
            index["trips"][trip_id]["name"] = f"{dest}{' · ' + dates if dates else ''}"
        save_user_index(user_token, index)
    except Exception:
        pass  # Never surface errors during capture


async def extract_metadata_background(user_token: str, trip_id: str):
    await asyncio.to_thread(_extract_metadata_sync, user_token, trip_id)


_IMAGE_SYSTEM = (
    "You extract travel-relevant details from images for a travel planning app. "
    "The image may be a screenshot of a booking confirmation, hotel, restaurant, attraction, "
    "map, itinerary, flight details, or any travel content. "
    "Only include fields with clearly visible values — never guess. "
    "status: 'extracted' = found useful info; 'low_confidence' = very little info. "
    "category: one of hotel | restaurant | flight | attraction | transport | other."
)


def _analyze_image_sync(user_token: str, trip_id: str, item_id: str, file_id: str, filename: str):
    """Analyze an uploaded image with Claude vision and store extracted travel details."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)

        response = client.beta.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=_IMAGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "file", "file_id": file_id}},
                    {"type": "text", "text": (
                        f"This is a travel image (filename: {filename}). "
                        "Extract all travel details visible in it — booking references, "
                        "dates, addresses, prices, names, and anything else useful for trip planning."
                    )},
                ],
            }],
            betas=["files-api-2025-04-14"],
            output_config={"format": {"type": "json_schema", "schema": _LINK_EXTRACTION_SCHEMA}},
        )

        ext = json.loads(response.content[0].text)

        lines = [f"Image: {filename}", f"Title: {ext['title']}", f"Type: {ext['category']}", f"Summary: {ext['summary']}"]
        for k in ["address", "city", "country", "price", "check_in", "check_out",
                  "hours", "phone", "cuisine", "airline", "flight_number",
                  "departure", "arrival", "rating", "notes"]:
            if ext.get(k):
                lines.append(f"{k.replace('_', ' ').title()}: {ext[k]}")

        _update_item(user_token, trip_id, item_id, {
            "extraction": ext,
            "content": "\n".join(lines),
        })

    except Exception:
        _update_item(user_token, trip_id, item_id, {
            "extraction": {"status": "failed", "title": filename, "category": "other", "summary": ""},
            "content": f"Image: {filename}\n[Analysis failed]",
        })


async def analyze_image_background(user_token: str, trip_id: str, item_id: str, file_id: str, filename: str):
    await asyncio.to_thread(_analyze_image_sync, user_token, trip_id, item_id, file_id, filename)


@app.post("/api/trips/{trip_id}/import-bookmarks")
async def import_bookmarks(
    request: Request,
    background_tasks: BackgroundTasks,
    trip_id: str,
    file: UploadFile = File(...),
):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    content = (await file.read()).decode("utf-8", errors="replace")

    # Chrome exports bookmarks as: <A HREF="url" ...>title</A>
    pattern = re.compile(r'<A\s+HREF="([^"]+)"[^>]*>([^<]*)</A>', re.IGNORECASE)
    matches = pattern.findall(content)
    bookmarks = [(url, title.strip()) for url, title in matches if url.lower().startswith("http")][:75]

    if not bookmarks:
        raise HTTPException(status_code=400, detail="No bookmarks found — make sure you exported from Chrome as HTML")

    for url, title in bookmarks:
        item = {
            "id": str(uuid.uuid4())[:8],
            "type": "link",
            "label": title or "",
            "url": url,
            "content": f"Source: {url}",
            "extraction": {"status": "pending", "title": title or _url_to_title(url), "category": "other", "summary": ""},
            "file_id": None,
        }
        index["trips"][trip_id]["items"].append(item)
        background_tasks.add_task(extract_link_background, token, trip_id, item["id"], url)

    index["trips"][trip_id]["itinerary"] = None
    save_user_index(token, index)
    return {"ok": True, "imported": len(bookmarks)}


@app.post("/api/trips/{trip_id}/upload")
async def upload_content(
    request: Request,
    background_tasks: BackgroundTasks,
    trip_id: str,
    content_type: str = Form(...),  # "text" | "file" | "email" | "link"
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    label: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
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
        file_kind = "document" if mime == "application/pdf" else "image"
        item["type"] = "file"
        item["file_kind"] = file_kind
        item["file_id"] = uploaded.id
        item["filename"] = file.filename
        item["mime"] = mime
        item["content"] = None
        item["extraction"] = {"status": "pending", "title": file.filename, "category": "other", "summary": ""}

    index["trips"][trip_id]["items"].append(item)
    index["trips"][trip_id]["itinerary"] = None
    save_user_index(token, index)

    # Background: silently parse metadata — never blocks the response
    background_tasks.add_task(extract_metadata_background, token, trip_id)
    if content_type == "link":
        background_tasks.add_task(extract_link_background, token, trip_id, item["id"], url)
    if item.get("file_kind") == "image":
        background_tasks.add_task(analyze_image_background, token, trip_id, item["id"], item["file_id"], file.filename)

    return {"ok": True, "item": item}


@app.post("/api/trips/{trip_id}/organize")
async def organize_trip(request: Request, trip_id: str):
    token = get_user_token(request)
    index = load_user_index(token)
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
    save_user_index(token, index)
    return {"itinerary": itinerary_text}


@app.post("/api/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    trip_id: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    history: str = Form(default="[]"),
):
    token = get_user_token(request)
    index = load_user_index(token)
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
    if not GMAIL_TOKEN_FILE.exists():
        return {"connected": False}
    try:
        service = get_gmail_service()
        # Make a lightweight real API call to confirm the token actually works
        service.users().getProfile(userId="me").execute()
        return {"connected": True}
    except Exception:
        GMAIL_TOKEN_FILE.unlink(missing_ok=True)
        return {"connected": False}


@app.post("/api/email/scan")
async def scan_emails(request: Request, trip_id: str = Form(...), keywords: str = Form(...)):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    service = get_gmail_service()

    # Build Gmail search query from comma-separated keywords
    terms = [k.strip() for k in keywords.split(",") if k.strip()]
    query = " OR ".join(terms)

    try:
        results = service.users().messages().list(
            userId="me", q=query, maxResults=25
        ).execute()
    except Exception as e:
        if "invalid_grant" in str(e) or "Token has been expired or revoked" in str(e):
            GMAIL_TOKEN_FILE.unlink(missing_ok=True)
            raise HTTPException(status_code=401, detail="Gmail session expired — please reconnect")
        raise HTTPException(status_code=502, detail=f"Gmail API error: {e}")

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
    request: Request,
    trip_id: str = Form(...),
    subject: str = Form(...),
    sender: str = Form(...),
    date: str = Form(...),
    body: str = Form(...),
):
    token = get_user_token(request)
    index = load_user_index(token)
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
    save_user_index(token, index)
    return {"ok": True, "item": item}


# ---------------------------------------------------------------------------
# PWA share target
# ---------------------------------------------------------------------------

@app.get("/share", response_class=HTMLResponse)
async def share_target(request: Request, url: Optional[str] = None, text: Optional[str] = None, title: Optional[str] = None):
    """Receives shared URLs from the Android share sheet (PWA share target)."""
    if not request.cookies.get("apollo_user"):
        return RedirectResponse("/")
    # Extract a URL from whichever param TikTok/other apps populated
    shared_url = url or ""
    if not shared_url and text:
        # Some apps put the URL in the text field
        match = re.search(r"https?://\S+", text)
        if match:
            shared_url = match.group(0)
    # Pass it to the app via a query param so JS can pre-fill the capture box
    return RedirectResponse(f"/?share={shared_url}" if shared_url else "/")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)

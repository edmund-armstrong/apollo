import os
import json
import uuid
import base64
import asyncio
import re
import logging
import urllib.parse
import hmac
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [gmaps] %(message)s")
_log = logging.getLogger("apollo.gmaps")
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
        data = json.loads(f.read_text())
        if "bank" not in data:
            data["bank"] = []
        return data
    return {"trips": {}, "bank": []}


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


def _fetch_page_data(url: str) -> tuple:
    """Fetch a URL and return (body_text, og_image_url). og_image_url may be ''."""
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
                html = resp.text
                # Extract og:image from <meta> tag (either attribute order)
                og_image = ""
                m = re.search(
                    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                    html, re.I,
                )
                if not m:
                    m = re.search(
                        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                        html, re.I,
                    )
                if m:
                    og_image = m.group(1).strip()
                parser = _TextExtractor()
                parser.feed(html)
                text = parser.get_text()
                if len(text.strip()) >= 80:
                    return text, og_image
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _url_to_title(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
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
        "status":           {"type": "string"},
        "title":            {"type": "string"},
        "category":         {"type": "string"},
        "card_description": {"type": "string"},
        "summary":          {"type": "string"},
        "details":          {"type": "array", "items": {"type": "string"}},
        "address":          {"type": "string"},
        "city":             {"type": "string"},
        "country":          {"type": "string"},
        "price":            {"type": "string"},
        "check_in":         {"type": "string"},
        "check_out":        {"type": "string"},
        "hours":            {"type": "string"},
        "phone":            {"type": "string"},
        "cuisine":          {"type": "string"},
        "airline":          {"type": "string"},
        "flight_number":    {"type": "string"},
        "departure":        {"type": "string"},
        "arrival":          {"type": "string"},
        "rating":           {"type": "string"},
        "notes":            {"type": "string"},
    },
    # output_config structured outputs require ALL properties in "required"
    "required": [
        "status", "title", "category", "card_description", "summary", "details",
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
    "category: one of hotel | restaurant | flight | attraction | transport | other. "
    "card_description: exactly 2 sentences, specific and informative — what is this and why would a traveler care? "
    "summary: 3–5 sentence paragraph expanding on what makes this worth visiting or reading. "
    "details: array of 2–4 short strings. Infer content type (restaurant/activity/article/hotel) and extract "
    "the most useful facts (e.g. neighborhood, price range, hours, must-try dish, key takeaway). "
    "Use empty string for unknown string fields, empty array for details if no useful facts found."
)


def _is_google_maps_url(url: str) -> bool:
    return bool(re.search(r"(maps\.google\.|google\.[a-z]+/maps|maps\.app\.goo\.gl|goo\.gl/maps)", url))


def _gmaps_base_result(title: str) -> dict:
    """Return a minimal GMaps extraction result (all required fields present)."""
    return {
        "status": "extracted",
        "title": title,
        "category": "attraction",
        "card_description": "",
        "summary": "",
        "details": [],
        "address": "",
        "city": "",
        "country": "",
        "price": "",
        "check_in": "",
        "check_out": "",
        "hours": "",
        "phone": "",
        "cuisine": "",
        "airline": "",
        "flight_number": "",
        "departure": "",
        "arrival": "",
        "rating": "",
        "notes": "",
    }


def _gmaps_type_label(types: list) -> str:
    """Map Google place types list to a human-readable label."""
    checks = [
        (["restaurant", "meal_delivery", "meal_takeaway"], "Restaurant"),
        (["cafe", "bakery"], "Café"),
        (["bar", "night_club"], "Bar"),
        (["lodging", "hotel"], "Hotel"),
        (["museum"], "Museum"),
        (["park", "national_park", "natural_feature"], "Park"),
        (["tourist_attraction", "amusement_park"], "Attraction"),
        (["shopping_mall", "store", "clothing_store", "department_store"], "Shopping"),
        (["spa", "beauty_salon"], "Spa"),
        (["gym", "health"], "Fitness"),
        (["airport", "transit_station", "train_station", "bus_station"], "Transit"),
    ]
    for type_list, label in checks:
        if any(t in types for t in type_list):
            return label
    return "Place"


def _get_place_details(place_id: str, api_key: str) -> Optional[dict]:
    """Fetch full place details from the Google Places API."""
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "fields": (
                        "name,formatted_address,address_components,rating,"
                        "user_ratings_total,opening_hours,formatted_phone_number,"
                        "website,types,price_level,editorial_summary,photos"
                    ),
                    "key": api_key,
                },
            )
        data = r.json()
        if data.get("status") != "OK":
            return None
        return data.get("result")
    except Exception:
        return None


def _generate_gmaps_summary(
    name: str, type_label: str, address: str,
    rating, review_count, price: Optional[str],
    hours: Optional[list], editorial: str,
) -> str:
    """Call Claude to generate a traveler-friendly summary for a saved place."""
    try:
        ant_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not ant_key:
            return editorial or ""
        client_ant = anthropic.Anthropic(api_key=ant_key)
        rating_str = (
            f"{rating} ({review_count:,} reviews)" if rating and review_count
            else str(rating) if rating else "Not available"
        )
        hours_str = "\n".join(hours) if hours else "Not available"
        prompt = (
            f"You are summarizing a place for a traveler who saved it to their trip.\n\n"
            f"Place name: {name}\nType: {type_label}\nAddress: {address}\n"
            f"Rating: {rating_str}\nPrice level: {price or 'Not specified'}\n"
            f"Hours:\n{hours_str}\n"
            f"Additional context: {editorial or 'None provided'}\n\n"
            f"Write a 2-4 sentence summary a traveler would find useful. Cover what makes this "
            f"place worth visiting, what to expect, and any practical context (neighborhood, vibe, "
            f"price point). Be specific, not generic.\n\nReturn only the summary text. No preamble."
        )
        resp = client_ant.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return editorial or ""


def _enrich_place_sync(place_id: str, api_key: str, fallback_title: str = "Place") -> dict:
    """
    Given a place_id, fetch Place Details + generate Claude summary.
    Returns a full extraction dict (same shape as _fetch_google_maps_place output).
    """
    place = _get_place_details(place_id, api_key)
    if not place:
        return _gmaps_base_result(fallback_title)

    _log.info("Place Details → name: %r  address: %r  types: %s",
              place.get("name"), place.get("formatted_address"), place.get("types", [])[:4])

    types = place.get("types", [])
    category = "attraction"
    if any(t in types for t in ["restaurant", "food", "cafe", "bar", "bakery", "meal_takeaway"]):
        category = "restaurant"
    elif any(t in types for t in ["lodging", "hotel"]):
        category = "hotel"
    elif any(t in types for t in ["airport", "transit_station", "bus_station", "train_station"]):
        category = "transport"
    type_label = _gmaps_type_label(types)

    weekday_text = (place.get("opening_hours") or {}).get("weekday_text") or []
    hours_text = " | ".join(weekday_text) if weekday_text else ""

    price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    price = price_map.get(place.get("price_level"), "")

    rating = place.get("rating")
    review_count = place.get("user_ratings_total") or 0
    rating_str = ""
    if rating and review_count:
        rating_str = f"{rating} ({review_count:,} reviews)"
    elif rating:
        rating_str = str(rating)

    neighborhood = ""
    for comp in (place.get("address_components") or []):
        comp_types = comp.get("types", [])
        if any(t in comp_types for t in ["neighborhood", "sublocality", "sublocality_level_1"]):
            neighborhood = comp["long_name"]
            break
    if not neighborhood:
        for comp in (place.get("address_components") or []):
            if "locality" in comp.get("types", []):
                neighborhood = comp["long_name"]
                break

    photo_urls = []
    for photo in (place.get("photos") or [])[:3]:
        ref = photo.get("photo_reference")
        if ref:
            photo_urls.append(
                f"https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=800&photo_reference={ref}&key={api_key}"
            )

    editorial = (place.get("editorial_summary") or {}).get("overview", "")
    summary = _generate_gmaps_summary(
        name=place.get("name", fallback_title),
        type_label=type_label,
        address=place.get("formatted_address", ""),
        rating=rating,
        review_count=review_count,
        price=price or None,
        hours=weekday_text,
        editorial=editorial,
    )

    details = []
    if neighborhood:
        details.append(f"Located in {neighborhood}")
    if price:
        details.append(f"Price: {price}")
    if type_label and type_label != "Place":
        details.append(type_label)
    if place.get("formatted_phone_number"):
        details.append(f"Phone: {place['formatted_phone_number']}")

    result = _gmaps_base_result(place.get("name", fallback_title))
    result.update({
        "category": category,
        "card_description": summary,
        "summary": summary,
        "details": details,
        "address": place.get("formatted_address", ""),
        "city": neighborhood,
        "price": price,
        "hours": hours_text,
        "phone": place.get("formatted_phone_number", ""),
        "cuisine": type_label,
        "rating": rating_str,
        "photo_urls": photo_urls,
        "raw_rating": rating or 0,
        "review_count": review_count,
        "neighborhood": neighborhood,
        "type_label": type_label,
        "weekday_text": weekday_text,
    })
    _log.info("Enrichment complete → title: %r  address: %r  photos: %d",
              result["title"], result["address"], len(photo_urls))
    return result


def _fetch_google_maps_place(url: str) -> dict:
    """
    Fetch rich place data from a Google Maps URL using Places API + Claude summary.
    Returns extraction dict with standard fields + extended GMaps fields.
    Logs each step to aid debugging wrong-place issues.
    """
    _log.info("=== GMaps extraction start ===")
    _log.info("Original URL: %s", url)

    # ── Step 1: Resolve shortened URLs (goo.gl, maps.app.goo.gl) ──────────────
    resolved = url
    if "goo.gl" in url:
        try:
            with httpx.Client(follow_redirects=True, timeout=10) as c:
                r = c.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resolved = str(r.url)
            _log.info("Resolved short URL → %s", resolved)
        except Exception as exc:
            _log.warning("Short-URL resolution failed: %s", exc)
    else:
        _log.info("No short URL, using as-is")

    # ── Step 2: Extract place identity from the resolved URL ───────────────────
    place_id: Optional[str] = None
    place_query: Optional[str] = None
    lat_lng: Optional[str] = None

    # 2a. Place ID embedded as !1sChIJ... in the path parameters
    m = re.search(r"!1s(ChIJ[A-Za-z0-9_\-]+)", resolved)
    if m:
        place_id = m.group(1)
        _log.info("Extracted Place ID from !1s pattern: %s", place_id)

    # 2b. Place name in path: /maps/place/Name+of+Place/
    if not place_id:
        m = re.search(r"/maps/place/([^/@?&]+)", resolved)
        if m:
            place_query = urllib.parse.unquote_plus(m.group(1))
            _log.info("Extracted place name from path: %r", place_query)

    # 2c. ?q= query string
    if not place_id and not place_query:
        m = re.search(r"[?&]q=([^&]+)", resolved)
        if m:
            place_query = urllib.parse.unquote_plus(m.group(1))
            _log.info("Extracted place name from ?q=: %r", place_query)

    # 2d. Coordinates @lat,lng for location bias
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", resolved)
    if m:
        lat_lng = f"{m.group(1)},{m.group(2)}"
        _log.info("Extracted coordinates for bias: %s", lat_lng)

    if not place_id and not place_query:
        _log.warning("Could not extract any place identity from URL")

    fallback_title = place_query or "Google Maps place"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        _log.warning("GOOGLE_PLACES_API_KEY not set — returning minimal result")
        return _gmaps_base_result(fallback_title)

    # ── Step 3: Get place_id if we don't already have one ─────────────────────
    if not place_id:
        search_input = place_query or resolved
        params: dict = {
            "input": search_input,
            "inputtype": "textquery",
            "fields": "place_id,name,formatted_address",
            "key": api_key,
        }
        # Add coordinate-based location bias to disambiguate common names
        if lat_lng:
            params["locationbias"] = f"point:{lat_lng}"
        _log.info("findplacefromtext query: %r  bias: %s", search_input, lat_lng or "none")
        try:
            with httpx.Client(timeout=10) as c:
                r = c.get(
                    "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                    params=params,
                )
            fp_data = r.json()
        except Exception as exc:
            _log.error("findplacefromtext request failed: %s", exc)
            return _gmaps_base_result(fallback_title)

        _log.info("findplacefromtext status: %s  candidates: %d",
                  fp_data.get("status"), len(fp_data.get("candidates", [])))
        for i, c in enumerate(fp_data.get("candidates", [])[:3]):
            _log.info("  candidate[%d]: place_id=%s  name=%s  address=%s",
                      i, c.get("place_id"), c.get("name"), c.get("formatted_address"))

        if fp_data.get("status") != "OK" or not fp_data.get("candidates"):
            _log.warning("findplacefromtext returned no usable candidates")
            return _gmaps_base_result(fallback_title)

        place_id = fp_data["candidates"][0]["place_id"]
        _log.info("Using place_id: %s", place_id)

    # ── Step 4: Enrich using shared helper ────────────────────────────────────
    enriched = _enrich_place_sync(place_id, api_key, fallback_title)
    _log.info("=== GMaps extraction end ===")
    return enriched


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

        # Google Maps: rich place card via Places API + Claude summary
        if _is_google_maps_url(url):
            try:
                ext = _fetch_google_maps_place(url)
                photo_urls = ext.get("photo_urls", [])
                lines = [f"Source: {url}", f"Title: {ext['title']}", f"Type: {ext['category']}"]
                if ext.get("summary"):
                    lines.append(f"Summary: {ext['summary']}")
                for k in ["address", "rating", "phone", "hours", "price"]:
                    if ext.get(k):
                        lines.append(f"{k.title()}: {ext[k]}")
                updates: dict = {
                    "extraction": ext,
                    "content": "\n".join(lines),
                    "is_gmaps": True,
                }
                if photo_urls:
                    updates["og_image"] = photo_urls[0]
                _update_item(user_token, trip_id, item_id, updates)
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

        og_image = ""
        try:
            if "reddit.com/" in url:
                page_text = _fetch_reddit_text_sync(url)
            else:
                page_text, og_image = _fetch_page_data(url)
        except Exception:
            page_text = None

        if not page_text or len(page_text.strip()) < 80:
            _update_item(user_token, trip_id, item_id, {
                "extraction": {
                    "status": "inaccessible",
                    "title": _url_to_title(url),
                    "category": "other",
                    "card_description": "",
                    "summary": "",
                    "details": [],
                },
                "content": f"Source: {url}\n[Page inaccessible — raw URL preserved]",
            })
            return

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=_LINK_SYSTEM,
            messages=[{"role": "user", "content":
                f"Extract travel details from this page.\nURL: {url}\n\nContent:\n{page_text[:8000]}"}],
            output_config={"format": {"type": "json_schema", "schema": _LINK_EXTRACTION_SCHEMA}},
        )

        ext = json.loads(response.content[0].text)

        # Build formatted content for AI context in chat / itinerary
        lines = [
            f"Source: {url}",
            f"Title: {ext['title']}",
            f"Type: {ext['category']}",
            f"Summary: {ext.get('summary', '')}",
        ]
        for k in ["address", "city", "country", "price", "check_in", "check_out",
                  "hours", "phone", "cuisine", "airline", "flight_number",
                  "departure", "arrival", "rating", "notes"]:
            if ext.get(k):
                lines.append(f"{k.replace('_', ' ').title()}: {ext[k]}")
        for d in (ext.get("details") or []):
            if d:
                lines.append(f"- {d}")
        if ext.get("status") == "inaccessible":
            lines.append("[Note: page is behind a paywall or login — consider pasting content manually]")
        elif ext.get("status") == "low_confidence":
            lines.append("[Note: limited info extracted — consider pasting page content manually]")

        updates: dict = {"extraction": ext, "content": "\n".join(lines)}
        if og_image:
            updates["og_image"] = og_image
        _update_item(user_token, trip_id, item_id, updates)

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


@app.patch("/api/trips/{trip_id}/items/{item_id}")
async def patch_item(
    request: Request,
    trip_id: str,
    item_id: str,
    user_notes: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
    trip = index["trips"].get(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    for item in trip["items"]:
        if item["id"] == item_id:
            if user_notes is not None:
                item["user_notes"] = user_notes.strip()
            save_user_index(token, index)
            return {"ok": True, "item": item}
    raise HTTPException(status_code=404, detail="Item not found")


# ---------------------------------------------------------------------------
# Travel Bank
# ---------------------------------------------------------------------------

@app.get("/api/bank")
async def get_bank(request: Request):
    token = get_user_token(request)
    index = load_user_index(token)
    # Collect all trip items annotated with trip info
    trip_items = []
    for trip_id, trip in index.get("trips", {}).items():
        for item in trip.get("items", []):
            trip_items.append({**item, "_trip_id": trip_id, "_trip_name": trip.get("name", "")})
    return {"bank_items": index.get("bank", []), "trip_items": trip_items}


@app.post("/api/bank/upload")
async def bank_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    content_type: str = Form(...),
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    label: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
    client = get_client()

    item = {"id": str(uuid.uuid4())[:8], "type": content_type, "label": label or "", "created_at": datetime.now().isoformat()}

    if content_type in ("text", "email"):
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

    index["bank"].append(item)
    save_user_index(token, index)

    if content_type == "link":
        background_tasks.add_task(_bank_extract_link_background, token, item["id"], url)

    return {"ok": True, "item": item}


async def _bank_extract_link_background(token: str, item_id: str, url: str):
    await asyncio.to_thread(_bank_extract_link_sync, token, item_id, url)


def _update_bank_item(token: str, item_id: str, updates: dict):
    index = load_user_index(token)
    for i, item in enumerate(index["bank"]):
        if item["id"] == item_id:
            index["bank"][i].update(updates)
            save_user_index(token, index)
            return


def _bank_extract_link_sync(token: str, item_id: str, url: str):
    """Run link extraction for a bank item by routing through _extract_link_sync
    via a temporary single-trip context, then moving the result to the bank."""
    # We attach the item to a temp slot in the index, run extraction, then move it back.
    # Simpler: just replicate the update pattern using _update_bank_item.
    try:
        if _is_google_maps_url(url):
            ext = _fetch_google_maps_place(url)
            photo_urls = ext.get("photo_urls", [])
            updates: dict = {"extraction": ext, "content": ext.get("summary", ""), "is_gmaps": True}
            if photo_urls:
                updates["og_image"] = photo_urls[0]
            _update_bank_item(token, item_id, updates)
            return

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
                _update_bank_item(token, item_id, {"extraction": ext, "content": f"Source: {url}"})
            except Exception:
                pass
            return

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return
        client = anthropic.Anthropic(api_key=api_key)

        og_image = ""
        try:
            page_text, og_image = _fetch_page_data(url)
        except Exception:
            page_text = None

        if not page_text or len(page_text.strip()) < 80:
            _update_bank_item(token, item_id, {
                "extraction": {"status": "inaccessible", "title": _url_to_title(url),
                               "category": "other", "card_description": "", "summary": "", "details": []},
                "content": f"Source: {url}",
            })
            return

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"URL: {url}\n\nContent:\n{page_text[:6000]}"}],
            system=_LINK_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _LINK_EXTRACTION_SCHEMA}},
        )
        ext = json.loads(response.content[0].text)
        ext["status"] = "extracted"
        lines = [f"Source: {url}", f"Title: {ext.get('title','')}", f"Summary: {ext.get('summary','')}"]
        updates = {"extraction": ext, "content": "\n".join(lines)}
        if og_image:
            updates["og_image"] = og_image
        _update_bank_item(token, item_id, updates)
    except Exception:
        pass


@app.post("/api/bank/add-place")
async def bank_add_place(
    request: Request,
    place_id: str = Form(...),
    place_name: Optional[str] = Form(None),
):
    token = get_user_token(request)
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google Places API key not configured")
    enriched = await asyncio.to_thread(_enrich_place_sync, place_id, api_key, place_name or "Place")
    photo_urls = enriched.get("photo_urls", [])
    item = {
        "id": str(uuid.uuid4()),
        "type": "link",
        "url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
        "extraction": enriched,
        "content": enriched.get("summary", ""),
        "is_gmaps": True,
        "created_at": datetime.now().isoformat(),
    }
    if photo_urls:
        item["og_image"] = photo_urls[0]
    index = load_user_index(token)
    index["bank"].append(item)
    save_user_index(token, index)
    return item


@app.delete("/api/bank/items/{item_id}")
async def delete_bank_item(request: Request, item_id: str):
    token = get_user_token(request)
    index = load_user_index(token)
    index["bank"] = [i for i in index["bank"] if i["id"] != item_id]
    save_user_index(token, index)
    return {"ok": True}


@app.post("/api/bank/items/{item_id}/assign")
async def assign_bank_item(request: Request, item_id: str, trip_id: str = Form(...)):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index.get("trips", {}):
        raise HTTPException(status_code=404, detail="Trip not found")
    item = next((i for i in index["bank"] if i["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    index["bank"] = [i for i in index["bank"] if i["id"] != item_id]
    item["trip_id"] = trip_id
    index["trips"][trip_id]["items"].append(item)
    save_user_index(token, index)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Mailgun inbound email → Travel Bank
# ---------------------------------------------------------------------------

EMAIL_ROUTES_FILE = STORAGE_DIR / "email_routes.json"


def _load_email_routes() -> dict:
    if EMAIL_ROUTES_FILE.exists():
        return json.loads(EMAIL_ROUTES_FILE.read_text())
    return {}


def _save_email_routes(routes: dict):
    EMAIL_ROUTES_FILE.write_text(json.dumps(routes, indent=2))


def _get_or_create_inbound_address(token: str) -> str:
    """Return the inbound email prefix for this user, creating one if needed."""
    routes = _load_email_routes()
    # Check if user already has an address
    for prefix, t in routes.items():
        if t == token:
            return prefix
    # Create a new short stable prefix from the token
    prefix = hashlib.sha256(token.encode()).hexdigest()[:12]
    routes[prefix] = token
    _save_email_routes(routes)
    return prefix


def _verify_mailgun_signature(api_key: str, timestamp: str, token: str, signature: str) -> bool:
    digest = hmac.new(
        key=api_key.encode("utf-8"),
        msg=f"{timestamp}{token}".encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, signature)


@app.get("/api/email/inbound-address")
async def get_inbound_address(request: Request):
    token = get_user_token(request)
    prefix = _get_or_create_inbound_address(token)
    domain = os.getenv("MAILGUN_INBOUND_DOMAIN", "")
    address = f"{prefix}@{domain}" if domain else None
    return {"address": address, "prefix": prefix, "domain": domain}


@app.post("/api/email/inbound")
async def email_inbound(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()

    # Verify Mailgun webhook signature
    mg_key = os.getenv("MAILGUN_API_KEY", "")
    if mg_key:
        ts = form.get("timestamp", "")
        tok = form.get("token", "")
        sig = form.get("signature", "")
        if not _verify_mailgun_signature(mg_key, ts, tok, sig):
            raise HTTPException(status_code=403, detail="Invalid signature")

    # Resolve recipient → user token
    recipient = form.get("recipient", "")
    prefix = recipient.split("@")[0].lower() if "@" in recipient else recipient
    routes = _load_email_routes()
    user_token = routes.get(prefix)
    if not user_token:
        # Unknown address — return 200 so Mailgun doesn't retry
        return {"ok": False, "reason": "unknown recipient"}

    subject = form.get("subject", "(no subject)")
    sender = form.get("sender") or form.get("from", "")
    body = form.get("stripped-text") or form.get("body-plain", "")
    body_html = form.get("body-html", "")

    background_tasks.add_task(
        _process_inbound_email, user_token, sender, subject, body, body_html
    )
    return {"ok": True}


def _process_inbound_email(
    user_token: str, sender: str, subject: str, body: str, body_html: str
):
    """Save email to Travel Bank and extract any URLs found in the body."""
    try:
        index = load_user_index(user_token)
        now = datetime.now().isoformat()

        # Save the email itself as a bank text item
        email_item = {
            "id": str(uuid.uuid4())[:8],
            "type": "email",
            "label": subject,
            "content": f"From: {sender}\nSubject: {subject}\n\n{body}".strip(),
            "extraction": {
                "status": "ok",
                "title": subject,
                "category": "other",
                "summary": body[:300] if body else "",
                "card_description": body[:200] if body else "",
                "details": [f"From: {sender}"] if sender else [],
                "address": "", "city": "", "country": "", "price": "",
                "check_in": "", "check_out": "", "hours": "", "phone": "",
                "cuisine": "", "airline": "", "flight_number": "",
                "departure": "", "arrival": "", "rating": "", "notes": "",
            },
            "created_at": now,
        }
        index["bank"].append(email_item)
        save_user_index(user_token, index)

        # Extract URLs from body and save each as a bank link item
        urls = re.findall(r'https?://[^\s\'"<>()]+', body or body_html)
        # Deduplicate, skip tracking/unsubscribe noise
        seen = set()
        skip_patterns = re.compile(
            r'(unsubscribe|tracking|pixel|click\.|open\.|beacon|mailchimp|sendgrid|mandrillapp)',
            re.I
        )
        for url in urls:
            url = url.rstrip(".,;)")
            if url in seen or skip_patterns.search(url):
                continue
            seen.add(url)
            link_item_id = str(uuid.uuid4())[:8]
            link_item = {
                "id": link_item_id,
                "type": "link",
                "label": "",
                "url": url,
                "content": f"Source: {url}",
                "extraction": {"status": "pending", "title": _url_to_title(url), "category": "other", "summary": ""},
                "created_at": now,
            }
            index = load_user_index(user_token)
            index["bank"].append(link_item)
            save_user_index(user_token, index)
            # Enrich each link
            _bank_extract_link_sync(user_token, link_item_id, url)
    except Exception as e:
        _log.error("inbound email processing error: %s", e)


# ---------------------------------------------------------------------------
# Agent research
# ---------------------------------------------------------------------------

_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["name", "city"],
                "additionalProperties": False,
            },
        },
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url":    {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["url", "reason"],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body":  {"type": "string"},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["places", "links", "notes"],
    "additionalProperties": False,
}


def _run_agent_sync(description: str, destination: str, start_date: str, end_date: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    date_str = ""
    if start_date and end_date:
        date_str = f"{start_date} to {end_date}"
    elif start_date:
        date_str = f"starting {start_date}"
    prompt = f"""You are a travel research agent. The user is planning the following trip:

Destination: {destination or "Unknown"}
Dates: {date_str or "Not specified"}
Description: {description or "No description provided."}

Your job is to compile a research set of exactly:
- 5 Places (specific venues worth visiting — restaurants, bars, attractions, neighborhoods)
- 5 Links (useful URLs — guides, articles, booking pages, event listings relevant to this trip)
- 5 Notes (short practical notes — packing tips, local context, logistics, cultural advice)

For Places: return the place name and city so it can be looked up via Google Places API.
For Links: return a real, specific URL and a reason it's relevant.
For Notes: write the full note text directly (2-4 sentences each). Title each note.

Tailor everything tightly to the trip description, dates, and destination.
Return only valid JSON."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": _AGENT_SCHEMA}},
    )
    return json.loads(response.content[0].text)


def _agent_find_place_id_sync(name: str, city: str, api_key: str) -> Optional[str]:
    """Text-search for a place by name+city and return its place_id, or None."""
    query = f"{name}, {city}" if city else name
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(
                "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                params={
                    "input": query,
                    "inputtype": "textquery",
                    "fields": "place_id,name",
                    "key": api_key,
                },
            )
        data = r.json()
        candidates = data.get("candidates") or []
        if data.get("status") == "OK" and candidates:
            return candidates[0]["place_id"]
    except Exception:
        pass
    return None


def _agent_save_place_sync(user_token: str, trip_id: str, place: dict) -> dict:
    """Resolve place_id via text search, enrich via Places API, save as gmaps item."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    name = place.get("name", "Place")
    city = place.get("city", "")
    place_id = _agent_find_place_id_sync(name, city, api_key) if api_key else None
    if place_id and api_key:
        enriched = _enrich_place_sync(place_id, api_key, name)
    else:
        # Fallback: minimal item without Places API data
        enriched = _gmaps_base_result(name)
        enriched["summary"] = f"{name} in {city}" if city else name
        enriched["card_description"] = enriched["summary"]
        enriched["city"] = city
    photo_urls = enriched.get("photo_urls", [])
    item = {
        "id": str(uuid.uuid4()),
        "trip_id": trip_id,
        "type": "link",
        "url": f"https://www.google.com/maps/search/?q={urllib.parse.quote(f'{name} {city}')}",
        "extraction": enriched,
        "content": enriched.get("summary", ""),
        "is_gmaps": True,
        "agent_saved": True,
        "created_at": datetime.now().isoformat(),
    }
    if photo_urls:
        item["og_image"] = photo_urls[0]
    index = load_user_index(user_token)
    index["trips"][trip_id]["items"].append(item)
    save_user_index(user_token, index)
    return item


def _agent_save_link_sync(user_token: str, trip_id: str, url: str) -> dict:
    """Create a stub link item then run the full link extraction pipeline on it."""
    item_id = str(uuid.uuid4())
    stub = {
        "id": item_id,
        "trip_id": trip_id,
        "type": "link",
        "url": url,
        "label": url,
        "extraction": {"status": "pending", "title": url},
        "content": url,
        "agent_saved": True,
        "created_at": datetime.now().isoformat(),
    }
    index = load_user_index(user_token)
    index["trips"][trip_id]["items"].append(stub)
    save_user_index(user_token, index)
    # Enrich synchronously (runs Claude + OG fetch)
    _extract_link_sync(user_token, trip_id, item_id, url)
    # Re-load to get the enriched item; also stamp agent_saved
    index = load_user_index(user_token)
    for item in index["trips"][trip_id]["items"]:
        if item["id"] == item_id:
            item["agent_saved"] = True
            save_user_index(user_token, index)
            return item
    return stub


def _agent_save_note_sync(user_token: str, trip_id: str, note: dict) -> dict:
    """Save an agent-written note directly."""
    item = {
        "id": str(uuid.uuid4()),
        "trip_id": trip_id,
        "type": "text",
        "label": note.get("title", "Note"),
        "content": note.get("body", ""),
        "extraction": {
            "status": "ok",
            "title": note.get("title", "Note"),
            "summary": note.get("body", ""),
            "category": "other",
            "address": "", "city": "", "country": "", "price": "",
            "check_in": "", "check_out": "", "hours": "", "phone": "",
            "cuisine": "", "airline": "", "flight_number": "",
            "departure": "", "arrival": "", "rating": "", "notes": "",
            "card_description": note.get("body", ""),
            "details": [],
        },
        "agent_saved": True,
        "created_at": datetime.now().isoformat(),
    }
    index = load_user_index(user_token)
    index["trips"][trip_id]["items"].append(item)
    save_user_index(user_token, index)
    return item


@app.get("/api/places/autocomplete")
async def places_autocomplete(request: Request, input: str = "", session_token: str = ""):
    get_user_token(request)  # auth check
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key or not input.strip():
        return {"predictions": []}
    params = {
        "input": input.strip(),
        "types": "establishment",
        "key": api_key,
    }
    if session_token:
        params["sessiontoken"] = session_token
    try:
        with httpx.Client(timeout=8) as c:
            r = c.get(
                "https://maps.googleapis.com/maps/api/place/autocomplete/json",
                params=params,
            )
        data = r.json()
    except Exception:
        return {"predictions": []}
    predictions = []
    for p in (data.get("predictions") or [])[:5]:
        predictions.append({
            "place_id": p.get("place_id", ""),
            "name": p.get("structured_formatting", {}).get("main_text", p.get("description", "")),
            "address": p.get("structured_formatting", {}).get("secondary_text", ""),
            "description": p.get("description", ""),
        })
    return {"predictions": predictions}


@app.post("/api/trips/{trip_id}/add-place")
async def add_place(
    request: Request,
    trip_id: str,
    place_id: str = Form(...),
    place_name: Optional[str] = Form(None),
):
    user_token = get_user_token(request)
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google Places API key not configured")

    enriched = await asyncio.to_thread(
        _enrich_place_sync, place_id, api_key, place_name or "Place"
    )
    photo_urls = enriched.get("photo_urls", [])

    item_id = str(uuid.uuid4())
    item = {
        "id": item_id,
        "trip_id": trip_id,
        "type": "link",
        "url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
        "extraction": enriched,
        "content": enriched.get("summary", ""),
        "is_gmaps": True,
        "created_at": datetime.now().isoformat(),
    }
    if photo_urls:
        item["og_image"] = photo_urls[0]

    index = load_user_index(user_token)
    if trip_id not in index.get("trips", {}):
        raise HTTPException(status_code=404, detail="Trip not found")
    index["trips"][trip_id]["items"].append(item)
    save_user_index(user_token, index)
    return item


@app.post("/api/trips/{trip_id}/agent-research")
async def agent_research(
    request: Request,
    trip_id: str,
    description: Optional[str] = Form(None),
    destination: Optional[str] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
):
    token = get_user_token(request)
    index = load_user_index(token)
    if trip_id not in index["trips"]:
        raise HTTPException(status_code=404, detail="Trip not found")

    desc = (description or "").strip()
    dest = (destination or "").strip()
    sd = (start_date or "").strip()
    ed = (end_date or "").strip()

    async def generate():
        try:
            trip_name = index["trips"][trip_id].get("name", dest or "your trip")
            yield f"event: step_start\ndata: {json.dumps({'message': f'✦ Starting research for {trip_name}…'})}\n\n"
            result = await asyncio.to_thread(_run_agent_sync, desc, dest, sd, ed)

            # ── Places ────────────────────────────────────────────────────────
            yield f"event: step_start\ndata: {json.dumps({'message': '📍 Finding places…'})}\n\n"
            for place in (result.get("places") or [])[:5]:
                item = await asyncio.to_thread(_agent_save_place_sync, token, trip_id, place)
                title = item.get("extraction", {}).get("title") or place.get("name", "Place")
                yield f"event: item_saved\ndata: {json.dumps({**item, '_feed_title': title})}\n\n"
                await asyncio.sleep(0.05)

            # ── Links ─────────────────────────────────────────────────────────
            yield f"event: step_start\ndata: {json.dumps({'message': '🔗 Finding links…'})}\n\n"
            for link in (result.get("links") or [])[:5]:
                url = (link.get("url") or "").strip()
                if not url:
                    continue
                item = await asyncio.to_thread(_agent_save_link_sync, token, trip_id, url)
                title = item.get("extraction", {}).get("title") or url
                yield f"event: item_saved\ndata: {json.dumps({**item, '_feed_title': title})}\n\n"
                await asyncio.sleep(0.05)

            # ── Notes ─────────────────────────────────────────────────────────
            yield f"event: step_start\ndata: {json.dumps({'message': '📝 Writing notes…'})}\n\n"
            for note in (result.get("notes") or [])[:5]:
                item = await asyncio.to_thread(_agent_save_note_sync, token, trip_id, note)
                title = note.get("title", "Note")
                yield f"event: item_saved\ndata: {json.dumps({**item, '_feed_title': title})}\n\n"
                await asyncio.sleep(0.05)

            total = len(result.get("places", [])) + len(result.get("links", [])) + len(result.get("notes", []))
            yield f"event: complete\ndata: {json.dumps({'message': f'✦ Research complete — {total} items saved to your trip'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
                    trip_ctx += f"\n\n[{item.get('label') or item.get('type', 'item')}]:\n{item['content']}"

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

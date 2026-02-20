# Apollo ✈️ — AI Travel Planner

Apollo is an AI-powered travel companion that organizes your travel inspiration into itineraries and answers questions about your plans on the go.

## Features

- **Travel Vault** — Upload notes, emails, and screenshots; Claude organizes them into a clean itinerary
- **Geo-aware Chat** — Ask questions about your trip with real-time location context

## Setup

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Run the app**
   ```bash
   python main.py
   ```
   Then open [http://localhost:8000](http://localhost:8000)

## Usage

1. **Vault tab** — Create a trip, then add notes, emails, or screenshots
2. **Itinerary tab** — Select your trip and click "Organize with AI"
3. **Chat tab** — Select a trip, enable location, and ask anything

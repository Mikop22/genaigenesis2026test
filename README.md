# Agent<sup>2</sup> 🏠

[![Built at GenAI Genesis 2026](https://img.shields.io/badge/GenAI_Genesis-2026-blue)](#)

**Agent<sup>2</sup>** is an agentic real estate agent that finds your next home entirely via text message. No apps, no accounts, no forms. Just text us, tell us what you need, and get matched listings sent directly to your phone, and our agent will message the sellers on the listings you like!

## Features ✨

- **Conversational Interface:** Communicate naturally via SMS, powered by Twilio webhooks.
- **AI Criteria Extraction:** Uses advanced LLMs (via `railtracks` and OpenAI-compatible endpoints) to extract structured search criteria (location, intent, max price, beds, baths, and any miscellaneous preferences like "lots of windows" or "close to a park") from your text messages.
- **FastAPI Backend:** High-performance, async backend to handle robust conversation state and API rendering.
- **Sleek Landing Page:** A modern, responsive frontend that showcases the product and lets users initiate texts with a single click.
- **Data Gathering:** Uses combinations of `Playwright`, `BeautifulSoup4`, and `pandas` to scrape, format, and push listings.

## AI Agent 🤖

The core AI agent is the **Build Search Criteria agent** (`app/agents/build_search_criteria.py`). It sits at the heart of the pipeline and converts a freeform conversation into a structured format that the listing search can act on.

### What it does

The agent reads a real estate intake conversation and extracts six key search parameters from the natural language. It uses a large language model (via `railtracks` with an OpenAI-compatible endpoint) and two Jinja2 prompt templates to give the model its instructions.

### Input

A single string containing the full conversation transcript — the back-and-forth between a real estate agent and a prospective buyer or renter. This can come from an SMS thread, an intake call transcript, or any freeform text.

```python
from app.agents.build_search_criteria import extract_search_criteria

transcript = """
Agent: Hi! Are you looking to rent or buy?
User: Rent.
Agent: What's your monthly budget?
User: Around $2,500, no more than $3,000.
Agent: How many bedrooms do you need?
User: At least one, two if possible.
Agent: Any preferred neighborhood?
User: Somewhere in Brooklyn or Manhattan.
Agent: Any other must-haves?
User: I'd love lots of windows, and it'd be great to be close to a park.
"""

criteria = extract_search_criteria(transcript)
```

### Output

A Python `dict` with exactly six keys:

| Key | Type | Description | Example |
|---|---|---|---|
| `location` | `str` | City or neighborhood extracted from the conversation | `"Brooklyn or Manhattan NY"` |
| `intent` | `str` | `"rent"` or `"buy"` | `"rent"` |
| `price_max` | `int \| str` | Maximum budget (number, or `""` if not mentioned) | `3000` |
| `beds_min` | `int \| str` | Minimum bedrooms (number, or `""` if not mentioned) | `1` |
| `baths_min` | `int \| str` | Minimum bathrooms (number, or `""` if not mentioned) | `""` |
| `misc_criteria` | `list[str]` | Any other preferences mentioned (empty list if none) | `["lots of windows", "close to a park"]` |

```python
# Example output for the transcript above
{
    "location": "Brooklyn or Manhattan NY",
    "intent": "rent",
    "price_max": 3000,
    "beds_min": 1,
    "baths_min": "",
    "misc_criteria": ["lots of windows", "close to a park"]
}
```

If the model returns unparseable output, the agent falls back to safe defaults (`intent="rent"`, all other fields `""`).

### Where it fits

This dict is passed directly to the Zillow scraper to build a search URL and retrieve matching listings, which are then sent back to the user over SMS.

```
SMS transcript → extract_search_criteria() → criteria dict → Zillow scraper → listings → SMS reply
```

See [`docs/TRANSCRIPT_TO_CRITERIA.md`](docs/TRANSCRIPT_TO_CRITERIA.md) for a deeper dive.

---

## How It Works 🛠️

1. **Text Us:** Send a quick text to our dedicated Twilio number.
2. **Tell Us What You Need:** Our AI will ask you a series of questions about your budget, preferred area, move-in date, and deal-breakers.
3. **Get Matches:** We parse your structured criteria, match you with the best available real estate listings, and text you the results directly—ready to view!

## Tech Stack 🚀

- **Backend:** Python 3, FastAPI, Uvicorn, AWS EC2
- **AI / NLP:** Context extraction using `railtracks` and custom LLMs (GPT-OSS) 
- **Communications:** Twilio SMS API, Nvidia PersonaPlex (Quantized to 4B parameters)
- **Frontend:** HTML5, CSS3 (Custom styles), Vanilla JS
- **Scraping / Data:** Playwright, BeautifulSoup4, Pandas, lxml

## Setup & Installation ⚙️

### 1. Clone the repository
```bash
git clone <repository-url>
cd genaigenesis2026
```

### 2. Install dependencies
Create a virtual environment and install the required Python packages:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Fill out the variables in `.env` with your actual credentials for Twilio and your configured LLM (OpenAI-compatible) platform.

### 4. Run the Application
Start the development server using Uvicorn:
```bash
uvicorn app.main:app --reload
```
The application, including both the backend API and the static frontend, will be served at `http://localhost:8000`.

---
*Built with ❤️ during the GenAI Genesis 2026 Hackathon.*

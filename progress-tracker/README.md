# Progress Tracker

A full-stack web app to track your progress through the AI Learning curriculum (Projects 1–5 + Capstone).

## Tech Stack

- **Backend**: Python FastAPI + SQLAlchemy + JWT auth
- **Frontend**: React + Vite + Tailwind CSS
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Infrastructure**: Docker Compose

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
cd progress-tracker
docker compose up --build
```

App available at http://localhost (frontend) and http://localhost:8000/docs (API docs).

### Option 2: Manual Setup

**Backend:**
```bash
cd progress-tracker/backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python -m app.seed             # Seed curriculum data
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd progress-tracker/frontend
npm install
npm run dev
```

## Features

- 🔐 JWT authentication (register/login/refresh)
- 📊 Dashboard with overall progress and streak tracking
- 📋 Project cards with visual progress bars
- ✅ Topic-level progress tracking (Not Started → In Progress → Completed)
- 🌙 Dark theme UI
- 🐳 Docker Compose for one-command deployment

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive Swagger docs.

## Running Tests

```bash
cd progress-tracker/backend
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

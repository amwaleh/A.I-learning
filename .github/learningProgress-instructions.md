
# AI Learning Progress Tracker

Build a full-stack web app that lets learners track their progress through the AI curriculum (Projects 1–5 + Capstone).

## Tech Stack

- **Backend**: Python (FastAPI)
- **Frontend**: React (Vite)
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Auth**: JWT-based authentication

## Core Features

### 1. User Authentication
- Sign up / log in / log out
- Password hashing and JWT token management
- User profile page (name, avatar, join date)

### 2. Curriculum Dashboard
- Display all 6 projects (Projects 1–5 + Capstone) as cards
- Each card shows: project title, description, topic count, and completion percentage
- Visual progress bar per project

### 3. Topic-Level Progress Tracking
- Expand a project to see its topics and sub-topics (sourced from `Learning_instruction.md`)
- Mark individual topics as: **Not Started**, **In Progress**, or **Completed**
- Auto-calculate project completion percentage from topic statuses

### 4. Overall Progress Summary
- Dashboard header showing overall curriculum completion (e.g., "42% complete")
- Track total topics completed vs. total topics across all projects
- Simple streak counter: consecutive days with at least one topic completed

## Data Model

- **User**: id, email, password_hash, display_name, created_at
- **Project**: id, number, title, description
- **Topic**: id, project_id, title, parent_topic_id (nullable, for sub-topics)
- **UserProgress**: id, user_id, topic_id, status (not_started | in_progress | completed), updated_at

## API Endpoints

- `POST /auth/register` — Create account
- `POST /auth/login` — Get JWT token
- `GET /projects` — List all projects with user's progress
- `GET /projects/{id}/topics` — List topics for a project
- `PATCH /progress/{topic_id}` — Update topic status
- `GET /dashboard` — Get overall progress summary

## Pages

1. **Login / Register** — Auth forms
2. **Dashboard** — Project cards with progress bars, overall stats
3. **Project Detail** — Expandable topic checklist with status toggles
4. **Profile** — User info and settings

## Infrastructure & Scalability

- **Alembic migrations** — Version-controlled schema changes; required for PostgreSQL in production
- **Docker Compose** — Single `docker compose up` for backend + frontend + database
- **Environment config** — `.env` files with `pydantic-settings` for secrets, DB URLs, JWT expiry; no hardcoded values
- **CORS configuration** — Explicit allowed origins on FastAPI so the React dev server can talk to the API

## Reliability

- **API input validation** — Pydantic models on all request/response endpoints
- **Rate limiting** — On `/auth/login` and `/auth/register` to prevent brute-force (use `slowapi`)
- **Refresh tokens** — Short-lived access tokens + long-lived refresh tokens to avoid frequent logouts
- **Database connection pooling** — SQLAlchemy async with connection pool for PostgreSQL in production
- **Error handling middleware** — Consistent JSON error responses with proper HTTP status codes

## Testing & Quality

- **Backend tests** — `pytest` + `httpx.AsyncClient` for API integration tests; seed DB with fixture data
- **Frontend tests** — React Testing Library for component tests on dashboard and progress toggles
- **CI pipeline** — GitHub Actions running lint (`ruff`/`eslint`) + tests on every push
- **Database seeding script** — Auto-populate projects and topics from `Learning_instruction.md` to keep curriculum in sync

## Developer Experience

- **API docs** — FastAPI auto-generates Swagger UI at `/docs`; add response models for clean documentation
- **Logging** — Structured logging with `structlog` or Python `logging` for request tracing
- **Hot reload** — FastAPI `--reload` + Vite HMR for fast development iteration

## Future Enhancements

- **Notes per topic** — Let users attach personal notes/links to each topic
- **Export progress** — Download progress as JSON/CSV for offline review
- **Admin role** — Allow an instructor to view all learners' progress in aggregate
- **WebSocket updates** — Real-time dashboard updates for multi-device usage
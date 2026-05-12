
# AI Learning Progress Tracker

Build a full-stack web app that lets learners track their progress through the AI curriculum (Projects 1–5 + Capstone). The app serves as both a **progress tracker** and an **interactive learning platform** — each topic contains rich educational content rendered inline.

## Tech Stack

- **Backend**: Python 3.11+ (FastAPI with async SQLAlchemy)
- **Frontend**: React 18 (Vite + Tailwind CSS + Lucide icons)
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Auth**: JWT-based authentication (access + refresh tokens)
- **Markdown**: react-markdown + remark-gfm for rendering topic content

## Core Features

### 1. User Authentication
- Sign up / log in / log out with email and password
- Password hashing with bcrypt (via passlib)
- JWT access tokens (short-lived, 30 min) + refresh tokens (long-lived, 7 days)
- **Auth responses MUST include the user object** — both `/auth/login` and `/auth/register` must return `{ access_token, refresh_token, token_type, user: { id, email, display_name, created_at } }` so the frontend can set auth state and redirect immediately
- Protected routes redirect unauthenticated users to login
- User profile page (name, email, join date)

### 2. Curriculum Dashboard
- Display all 6 projects (Projects 1–5 + Capstone) as clickable cards
- Each card shows: project number, title, description, topic count, completed count, and completion percentage
- Visual progress bar per project with color coding (red → amber → green)
- Overall curriculum completion header with percentage and streak counter

### 3. Topic-Level Progress Tracking
- Expand a project to see its topics and sub-topics in a tree structure
- Mark individual topics as: **Not Started**, **In Progress**, or **Completed** via status icon click or dropdown
- Auto-calculate project completion percentage from topic statuses
- **Visual indicators**: Circle (not started), Clock (in progress), CheckCircle (completed) with color coding

### 4. Topic Learning Content ⭐ NEW
- **Every topic and sub-topic MUST have rich markdown learning content** (200–500 words each)
- Content is stored in the `content` TEXT column on the Topic model
- Clicking a topic title opens a content panel below the topic list
- Content is rendered as formatted markdown with:
  - Syntax-highlighted code blocks (Python examples)
  - Tables, bullet lists, and ASCII diagrams
  - Headers, bold/italic formatting
  - GFM (GitHub Flavored Markdown) support
- Content panel has a close button and highlights the selected topic
- Use `@tailwindcss/typography` plugin for prose styling in dark theme

### 5. Overall Progress Summary
- Dashboard header showing overall curriculum completion (e.g., "42% complete")
- Track total topics completed vs. total topics across all projects
- Simple streak counter: consecutive days with at least one topic completed
- Per-project breakdown with progress bars

## Data Model

- **User**: id (UUID), email (unique), password_hash, display_name, created_at
- **Project**: id, number (unique), title, description
- **Topic**: id, project_id (FK), title, **content (TEXT, nullable)**, parent_topic_id (nullable FK for sub-topics)
- **UserProgress**: id, user_id (FK), topic_id (FK), status (enum: not_started | in_progress | completed), updated_at
  - Unique constraint on (user_id, topic_id)

### Schema Consistency Rules
> These rules prevent frontend/backend field name mismatches — a common source of bugs:

- **TopicResponse** uses `children` (not `sub_topics`) for nested topics
- **ProjectListItem** uses `progress_percentage` (not `percentage`)
- **GET /projects/{id}/topics** returns `ProjectDetailResponse` object (with `.topics` field), NOT a bare list
- Frontend must access `response.data.topics` to get the topics array
- All response models must use `model_config = {"from_attributes": True}` for ORM compatibility

## API Endpoints

| Method | Path | Description | Auth | Returns |
|--------|------|-------------|------|---------|
| POST | `/auth/register` | Create account | No | TokenResponse (with user) |
| POST | `/auth/login` | Get JWT tokens | No | TokenResponse (with user) |
| POST | `/auth/refresh` | Refresh access token | No | AccessTokenResponse |
| GET | `/projects` | List all projects with progress | Yes | ProjectListItem[] |
| GET | `/projects/{id}/topics` | Topic tree for a project | Yes | ProjectDetailResponse |
| GET | `/projects/topics/{topic_id}` | Single topic with content | Yes | TopicResponse |
| PATCH | `/progress/{topic_id}` | Update topic status | Yes | ProgressResponse |
| GET | `/dashboard` | Overall progress summary | Yes | DashboardResponse |

### API Response Contract
```json
// TokenResponse (login/register)
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "display_name": "John",
    "created_at": "2026-01-01T00:00:00Z"
  }
}

// ProjectDetailResponse (GET /projects/{id}/topics)
{
  "id": 1,
  "number": 1,
  "title": "Build an LLM Playground",
  "description": "...",
  "topics": [
    {
      "id": 1,
      "title": "Pre-Training",
      "parent_topic_id": null,
      "status": "not_started",
      "content": "## Pre-Training\n\n...",
      "children": [
        {
          "id": 2,
          "title": "Data collection",
          "parent_topic_id": 1,
          "status": "in_progress",
          "content": "## Data Collection\n\n...",
          "children": []
        }
      ]
    }
  ]
}
```

## Pages

1. **Login / Register** — Auth forms with validation, redirects to dashboard on success
2. **Dashboard** — Project cards with progress bars, overall stats header, streak counter
3. **Project Detail** — Expandable topic tree with status toggles + **content panel** for selected topic
4. **Profile** — User info and settings

### Frontend Routing
- `/login` — Login form (redirects to `/` if authenticated)
- `/register` — Register form (redirects to `/` if authenticated)
- `/` — Dashboard (protected)
- `/projects/:id` — Project detail (protected)
- `/profile` — User profile (protected)

## Infrastructure & Scalability

- **Alembic migrations** — Version-controlled schema changes; required for PostgreSQL in production
- **Docker Compose** — Single `docker compose up` for backend + frontend + database
- **Environment config** — `.env` files with `pydantic-settings` for secrets, DB URLs, JWT expiry; no hardcoded values
- **CORS configuration** — Explicit allowed origins on FastAPI so the React dev server (port 5173) can talk to the API (port 8000)

## Reliability

- **API input validation** — Pydantic models on all request/response endpoints
- **Rate limiting** — On `/auth/login` (5/min) and `/auth/register` (3/min) to prevent brute-force (use `slowapi`)
- **Refresh tokens** — Short-lived access tokens (30 min) + long-lived refresh tokens (7 days) to avoid frequent logouts
- **Axios interceptors** — Frontend automatically refreshes expired access tokens using the refresh token
- **Database connection pooling** — SQLAlchemy async with connection pool for PostgreSQL in production
- **Error handling middleware** — Consistent JSON error responses with proper HTTP status codes

## Testing & Quality

- **Backend tests** — `pytest` + `httpx.AsyncClient` for API integration tests; seed DB with fixture data
- **Frontend tests** — React Testing Library for component tests on dashboard and progress toggles
- **CI pipeline** — GitHub Actions running lint (`ruff`/`eslint`) + tests on every push
- **Database seeding script** — `python -m app.seed` auto-populates projects, topics, AND topic content from the curriculum

## Seeding Requirements

The seed script (`app/seed.py`) must:
1. Create all 6 projects with titles and descriptions
2. Create parent topics and child topics with the correct `parent_topic_id` relationships
3. **Populate the `content` field** for every topic with rich educational markdown (200–500 words)
4. Content should include: concept explanations, key bullet points, Python code examples, ASCII diagrams or tables
5. Skip seeding if data already exists (idempotent)
6. The CURRICULUM data structure uses dicts for children (not plain strings):
```python
{
    "title": "Pre-Training",
    "content": "## Pre-Training\n\nPre-training is the foundational phase...",
    "children": [
        {"title": "Data collection", "content": "## Data Collection\n\n..."},
        {"title": "Data cleaning", "content": "## Data Cleaning\n\n..."},
    ]
}
```

## Developer Experience

- **API docs** — FastAPI auto-generates Swagger UI at `/docs`; add response models for clean documentation
- **Logging** — Structured logging with Python `logging` for request tracing
- **Hot reload** — FastAPI `--reload` + Vite HMR for fast development iteration
- **Database reset** — Delete `app.db` and re-run `python -m app.seed` to reset all data

## Common Pitfalls to Avoid

> These are bugs that were encountered during initial development:

1. **Login not redirecting**: Auth endpoints MUST return the user object in the token response. The frontend sets `isAuthenticated` based on the user object — if it's missing, the user stays on the login page.
2. **Topics not displaying**: The `/projects/{id}/topics` endpoint returns a `ProjectDetailResponse` object, not an array. Frontend must access `.data.topics` not `.data` directly.
3. **Field name mismatches**: Backend uses `children` and `progress_percentage`; frontend must match these exactly (not `sub_topics` or `percentage`).
4. **bcrypt warning**: `AttributeError: module 'bcrypt' has no attribute '__about__'` is harmless — it's a passlib compatibility issue with newer bcrypt versions. Does not affect functionality.
5. **Rate limiting on login**: 5 requests/minute limit — users may hit 429 during testing. Consider higher limits in development.

## Future Enhancements

- **Notes per topic** — Let users attach personal notes/links to each topic
- **Export progress** — Download progress as JSON/CSV for offline review
- **Admin role** — Allow an instructor to view all learners' progress in aggregate
- **WebSocket updates** — Real-time dashboard updates for multi-device usage
- **Search topics** — Full-text search across topic titles and content
- **Bookmarks** — Bookmark topics for quick access
- **Dark/Light theme toggle** — User preference for theme
- **Mobile responsive** — Optimized layout for mobile screens
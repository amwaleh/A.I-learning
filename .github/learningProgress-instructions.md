
# AI Learning Progress Tracker

Full-stack web app for tracking progress through the AI curriculum (Projects 1–5 + Capstone) with rich inline learning content.

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / async SQLAlchemy / SQLite (dev) / PostgreSQL (prod)
- **Frontend**: React 18 / Vite / Tailwind CSS / Lucide icons / react-markdown + remark-gfm
- **Auth**: JWT (access 30 min + refresh 7 days) via python-jose + passlib/bcrypt

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (React + Vite)"]
        UI[Pages & Components]
        Auth[AuthContext]
        API[Axios Client]
    end

    subgraph Backend["Backend (FastAPI)"]
        Routes[Routers]
        Models[SQLAlchemy Models]
        Schemas[Pydantic Schemas]
        Seed[Seed Script]
    end

    DB[(SQLite / PostgreSQL)]

    UI --> Auth
    UI --> API
    API -->|HTTP :8000| Routes
    Routes --> Models
    Routes --> Schemas
    Models --> DB
    Seed --> Models
```

## Auth Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend

    U->>F: Submit login form
    F->>B: POST /auth/login
    B-->>F: TokenResponse {access_token, refresh_token, user}
    F->>F: Store tokens, set isAuthenticated
    F-->>U: Redirect to Dashboard

    Note over F,B: On token expiry
    F->>B: POST /auth/refresh {refresh_token}
    B-->>F: New access_token
```

## User Workflow

```mermaid
flowchart LR
    A[Register/Login] --> B[Dashboard]
    B --> C[Select Project]
    C --> D[Browse Topics in Sidebar]
    D --> E[Read Content]
    D --> F[Toggle Status]
    F --> G[Progress Updates]
    G --> B
```

## Core Features

1. **Authentication** — Register/login with JWT. Responses MUST include user object in `TokenResponse`.
2. **Dashboard** — Project cards with progress bars, overall completion %, streak counter.
3. **Topic Progress** — Tree of topics/sub-topics. Status cycle: Not Started → In Progress → Completed.
4. **Topic Content** — Every topic has 200–500 words of markdown content. Split-panel layout (see `docs/ui-decisions.md`).
5. **Progress Summary** — Dashboard header with overall % and per-project breakdown.

## Data Model

```mermaid
erDiagram
    User ||--o{ UserProgress : tracks
    Project ||--o{ Topic : contains
    Topic ||--o{ Topic : "parent → children"
    Topic ||--o{ UserProgress : "tracked by"

    User {
        UUID id PK
        string email UK
        string password_hash
        string display_name
    }
    Project {
        int id PK
        int number UK
        string title
        string description
    }
    Topic {
        int id PK
        int project_id FK
        string title
        text content
        int parent_topic_id FK
    }
    UserProgress {
        int id PK
        UUID user_id FK
        int topic_id FK
        enum status
        datetime updated_at
    }
```

### Schema Rules
- `TopicResponse` uses `children` (not `sub_topics`)
- `ProjectListItem` uses `progress_percentage` (not `percentage`)
- `GET /projects/{id}/topics` returns `ProjectDetailResponse` object — access `.data.topics`
- All models: `model_config = {"from_attributes": True}`

## API Endpoints

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| POST | `/auth/register` | No | TokenResponse (with user) |
| POST | `/auth/login` | No | TokenResponse (with user) |
| POST | `/auth/refresh` | No | AccessTokenResponse |
| GET | `/projects` | Yes | ProjectListItem[] |
| GET | `/projects/{id}/topics` | Yes | ProjectDetailResponse |
| GET | `/projects/topics/{topic_id}` | Yes | TopicResponse |
| PATCH | `/progress/{topic_id}` | Yes | ProgressResponse |
| GET | `/dashboard` | Yes | DashboardResponse |

## Routes

`/login` · `/register` · `/` (dashboard) · `/projects/:id` · `/profile` — all protected except auth pages.

## Seeding

`python -m app.seed` — Creates projects, topics with hierarchy, populates `content`. Idempotent. Content in `app/content/project1.py`–`project6.py`.

## Common Pitfalls

1. **Login not redirecting** — Auth endpoints MUST return user object in TokenResponse.
2. **Topics not displaying** — Use `response.data.topics` not `response.data`.
3. **Field mismatches** — `children` not `sub_topics`, `progress_percentage` not `percentage`.
4. **bcrypt warning** — `'__about__'` AttributeError is harmless.
5. **Rate limiting** — Login: 5/min, Register: 3/min.

## UI Change Tracking ⚠️ MANDATORY

> **Every UI/layout change MUST update `docs/ui-decisions.md` to reflect the current state.**

Before changing any component layout/styling: read `docs/ui-decisions.md`.
After: update the relevant section. No changelog needed — git history tracks that.

## Infrastructure

- Alembic migrations for prod schema changes
- Docker Compose for full stack
- `.env` via pydantic-settings (no hardcoded secrets)
- CORS configured for dev (port 5173 → 8000)
- `run.ps1` script to start both servers

## Testing

- Backend: `pytest` + `httpx.AsyncClient`
- Frontend: React Testing Library
- CI: GitHub Actions (ruff + eslint + tests)
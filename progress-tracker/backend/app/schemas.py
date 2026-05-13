from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse | None" = None


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Projects & Topics ─────────────────────────────────────────────────
class TopicResponse(BaseModel):
    id: int
    title: str
    parent_topic_id: int | None = None
    status: str = "not_started"
    content: str | None = None
    children: list["TopicResponse"] = []

    model_config = {"from_attributes": True}


class ProjectListItem(BaseModel):
    id: int
    number: int
    title: str
    description: str | None = None
    total_topics: int = 0
    completed_topics: int = 0
    progress_percentage: float = 0.0

    model_config = {"from_attributes": True}


class ProjectDetailResponse(BaseModel):
    id: int
    number: int
    title: str
    description: str | None = None
    topics: list[TopicResponse] = []

    model_config = {"from_attributes": True}


# ── Progress ──────────────────────────────────────────────────────────
class ProgressUpdateRequest(BaseModel):
    status: str = Field(pattern=r"^(not_started|in_progress|completed)$")


class ProgressResponse(BaseModel):
    id: int
    topic_id: int
    status: str
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Dashboard ─────────────────────────────────────────────────────────
class DashboardResponse(BaseModel):
    total_topics: int
    completed_topics: int
    in_progress_topics: int
    overall_percentage: float
    streak_days: int
    projects: list[ProjectListItem] = []


# ── Checkpoints ───────────────────────────────────────────────────────
class CheckpointResponse(BaseModel):
    checkpoint_number: int
    label: str
    confirmed: bool = False
    confirmed_at: datetime | None = None


class TopicProgressStatus(BaseModel):
    scroll_percent: int
    time_spent: int
    checkpoints_total: int
    checkpoints_confirmed: int
    status: str


class AutoProgressRequest(BaseModel):
    scroll_percent: int = Field(ge=0, le=100)
    time_spent: int = Field(ge=0, description="Seconds spent on topic")

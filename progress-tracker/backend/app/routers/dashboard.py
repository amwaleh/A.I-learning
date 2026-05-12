from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Project, Topic, UserProgress, User, ProgressStatus
from app.schemas import DashboardResponse, ProjectListItem

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _compute_streak(dates: list[datetime]) -> int:
    """Compute the number of consecutive days ending today (or yesterday)."""
    if not dates:
        return 0

    unique_days = sorted({d.date() for d in dates}, reverse=True)
    today = datetime.now(timezone.utc).date()

    # Streak must include today or yesterday
    if unique_days[0] != today and unique_days[0] != today - timedelta(days=1):
        return 0

    streak = 1
    for i in range(1, len(unique_days)):
        if unique_days[i - 1] - unique_days[i] == timedelta(days=1):
            streak += 1
        else:
            break
    return streak


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Total topics
    total_result = await db.execute(select(func.count(Topic.id)))
    total_topics = total_result.scalar() or 0

    # Completed topics for this user
    completed_result = await db.execute(
        select(func.count(UserProgress.id)).where(
            UserProgress.user_id == user.id,
            UserProgress.status == ProgressStatus.completed,
        )
    )
    completed_topics = completed_result.scalar() or 0

    # In-progress topics
    in_progress_result = await db.execute(
        select(func.count(UserProgress.id)).where(
            UserProgress.user_id == user.id,
            UserProgress.status == ProgressStatus.in_progress,
        )
    )
    in_progress_topics = in_progress_result.scalar() or 0

    overall_pct = (
        round((completed_topics / total_topics) * 100, 1) if total_topics > 0 else 0.0
    )

    # Streak calculation based on progress update dates
    dates_result = await db.execute(
        select(UserProgress.updated_at).where(
            UserProgress.user_id == user.id,
            UserProgress.status.in_([ProgressStatus.in_progress, ProgressStatus.completed]),
        )
    )
    dates = [row[0] for row in dates_result.all() if row[0] is not None]
    streak = _compute_streak(dates)

    # Per-project breakdown
    projects_result = await db.execute(select(Project).order_by(Project.number))
    projects = projects_result.scalars().all()

    project_items = []
    for project in projects:
        topic_result = await db.execute(
            select(Topic).where(Topic.project_id == project.id)
        )
        all_topics = topic_result.scalars().all()
        total = len(all_topics)

        proj_completed_result = await db.execute(
            select(func.count(UserProgress.id)).where(
                UserProgress.user_id == user.id,
                UserProgress.topic_id.in_([t.id for t in all_topics]),
                UserProgress.status == ProgressStatus.completed,
            )
        )
        proj_completed = proj_completed_result.scalar() or 0
        pct = round((proj_completed / total) * 100, 1) if total > 0 else 0.0

        project_items.append(
            ProjectListItem(
                id=project.id,
                number=project.number,
                title=project.title,
                description=project.description,
                total_topics=total,
                completed_topics=proj_completed,
                progress_percentage=pct,
            )
        )

    return DashboardResponse(
        total_topics=total_topics,
        completed_topics=completed_topics,
        in_progress_topics=in_progress_topics,
        overall_percentage=overall_pct,
        streak_days=streak,
        projects=project_items,
    )

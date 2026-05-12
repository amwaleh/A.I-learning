from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Project, Topic, UserProgress, User, ProgressStatus
from app.schemas import ProjectListItem, ProjectDetailResponse, TopicResponse

router = APIRouter(prefix="/projects", tags=["projects"])


def _build_topic_tree(
    topics: list[Topic],
    progress_map: dict[int, str],
    parent_id: int | None = None,
) -> list[TopicResponse]:
    result = []
    for t in topics:
        actual_parent = t.parent_topic_id
        if actual_parent == parent_id:
            children = _build_topic_tree(topics, progress_map, parent_id=t.id)
            result.append(
                TopicResponse(
                    id=t.id,
                    title=t.title,
                    parent_topic_id=t.parent_topic_id,
                    status=progress_map.get(t.id, "not_started"),
                    content=t.content,
                    children=children,
                )
            )
    return result


@router.get("", response_model=list[ProjectListItem])
async def list_projects(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    projects_result = await db.execute(
        select(Project).order_by(Project.number)
    )
    projects = projects_result.scalars().all()

    items = []
    for project in projects:
        # Count all leaf topics for this project
        topic_result = await db.execute(
            select(Topic).where(Topic.project_id == project.id)
        )
        all_topics = topic_result.scalars().all()
        total = len(all_topics)

        # Count completed
        completed_result = await db.execute(
            select(func.count(UserProgress.id)).where(
                UserProgress.user_id == user.id,
                UserProgress.topic_id.in_([t.id for t in all_topics]),
                UserProgress.status == ProgressStatus.completed,
            )
        )
        completed = completed_result.scalar() or 0

        pct = round((completed / total) * 100, 1) if total > 0 else 0.0
        items.append(
            ProjectListItem(
                id=project.id,
                number=project.number,
                title=project.title,
                description=project.description,
                total_topics=total,
                completed_topics=completed,
                progress_percentage=pct,
            )
        )
    return items


@router.get("/{project_id}/topics", response_model=ProjectDetailResponse)
async def get_project_topics(
    project_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    topics_result = await db.execute(
        select(Topic).where(Topic.project_id == project_id).order_by(Topic.id)
    )
    all_topics = topics_result.scalars().all()
    topic_ids = [t.id for t in all_topics]

    progress_result = await db.execute(
        select(UserProgress).where(
            UserProgress.user_id == user.id,
            UserProgress.topic_id.in_(topic_ids),
        )
    )
    progress_map = {p.topic_id: p.status.value for p in progress_result.scalars().all()}

    tree = _build_topic_tree(all_topics, progress_map, parent_id=None)

    return ProjectDetailResponse(
        id=project.id,
        number=project.number,
        title=project.title,
        description=project.description,
        topics=tree,
    )


@router.get("/topics/{topic_id}")
async def get_topic_detail(
    topic_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    # Get user progress
    progress_result = await db.execute(
        select(UserProgress).where(
            UserProgress.user_id == user.id,
            UserProgress.topic_id == topic_id,
        )
    )
    progress = progress_result.scalar_one_or_none()
    status = progress.status.value if progress else "not_started"

    return TopicResponse(
        id=topic.id,
        title=topic.title,
        parent_topic_id=topic.parent_topic_id,
        status=status,
        content=topic.content,
        children=[],
    )

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Topic, UserProgress, User, ProgressStatus
from app.schemas import ProgressUpdateRequest, ProgressResponse

router = APIRouter(prefix="/progress", tags=["progress"])


@router.patch("/{topic_id}", response_model=ProgressResponse)
async def update_progress(
    topic_id: int,
    body: ProgressUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify topic exists
    topic_result = await db.execute(select(Topic).where(Topic.id == topic_id))
    if not topic_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    # Upsert user progress
    result = await db.execute(
        select(UserProgress).where(
            UserProgress.user_id == user.id,
            UserProgress.topic_id == topic_id,
        )
    )
    progress = result.scalar_one_or_none()

    new_status = ProgressStatus(body.status)

    if progress:
        progress.status = new_status
        progress.updated_at = datetime.now(timezone.utc)
    else:
        progress = UserProgress(
            user_id=user.id,
            topic_id=topic_id,
            status=new_status,
            updated_at=datetime.now(timezone.utc),
        )
        db.add(progress)

    await db.flush()
    await db.refresh(progress)

    return ProgressResponse(
        id=progress.id,
        topic_id=progress.topic_id,
        status=progress.status.value,
        updated_at=progress.updated_at,
    )

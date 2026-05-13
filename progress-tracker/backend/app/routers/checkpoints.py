from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.dependencies import get_current_user
from app.models import (
    Topic, TopicCheckpoint, UserCheckpoint, UserProgress, User, ProgressStatus
)
from app.schemas import CheckpointResponse, AutoProgressRequest, TopicProgressStatus

router = APIRouter(prefix="/projects/topics", tags=["checkpoints"])


@router.get("/{topic_id}/checkpoints", response_model=list[CheckpointResponse])
async def get_topic_checkpoints(
    topic_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all checkpoints for a topic with user's confirmation status."""
    topic_result = await db.execute(select(Topic).where(Topic.id == topic_id))
    if not topic_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Topic not found")

    checkpoints_result = await db.execute(
        select(TopicCheckpoint)
        .where(TopicCheckpoint.topic_id == topic_id)
        .order_by(TopicCheckpoint.checkpoint_number)
    )
    checkpoints = checkpoints_result.scalars().all()

    # Get user confirmations
    confirmed_result = await db.execute(
        select(UserCheckpoint).where(
            UserCheckpoint.user_id == user.id,
            UserCheckpoint.topic_id == topic_id,
        )
    )
    confirmed_map = {
        uc.checkpoint_number: uc.confirmed_at
        for uc in confirmed_result.scalars().all()
    }

    return [
        CheckpointResponse(
            checkpoint_number=cp.checkpoint_number,
            label=cp.label,
            confirmed=cp.checkpoint_number in confirmed_map,
            confirmed_at=confirmed_map.get(cp.checkpoint_number),
        )
        for cp in checkpoints
    ]


@router.post("/{topic_id}/checkpoints/{checkpoint_number}/confirm")
async def confirm_checkpoint(
    topic_id: int,
    checkpoint_number: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm a specific checkpoint for the current user."""
    # Verify checkpoint exists
    cp_result = await db.execute(
        select(TopicCheckpoint).where(
            TopicCheckpoint.topic_id == topic_id,
            TopicCheckpoint.checkpoint_number == checkpoint_number,
        )
    )
    if not cp_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Check if already confirmed
    existing = await db.execute(
        select(UserCheckpoint).where(
            UserCheckpoint.user_id == user.id,
            UserCheckpoint.topic_id == topic_id,
            UserCheckpoint.checkpoint_number == checkpoint_number,
        )
    )
    if existing.scalar_one_or_none():
        return {"status": "already_confirmed"}

    # Create confirmation
    uc = UserCheckpoint(
        user_id=user.id,
        topic_id=topic_id,
        checkpoint_number=checkpoint_number,
    )
    db.add(uc)
    await db.flush()

    return {"status": "confirmed", "checkpoint_number": checkpoint_number}


@router.patch("/{topic_id}/auto-progress", response_model=TopicProgressStatus)
async def auto_progress(
    topic_id: int,
    req: AutoProgressRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Auto-update progress based on scroll depth and time spent.
    Rules:
    - Opening (any scroll) → in_progress
    - scroll ≥ 90% + time ≥ 30s + all checkpoints confirmed → completed
    """
    # Get or create progress record
    progress_result = await db.execute(
        select(UserProgress).where(
            UserProgress.user_id == user.id,
            UserProgress.topic_id == topic_id,
        )
    )
    progress = progress_result.scalar_one_or_none()

    if not progress:
        progress = UserProgress(
            user_id=user.id,
            topic_id=topic_id,
            status=ProgressStatus.not_started,
        )
        db.add(progress)
        await db.flush()

    # Don't go backward
    if progress.status == ProgressStatus.completed:
        # Get checkpoint counts for response
        total_cps, confirmed_cps = await _get_checkpoint_counts(db, user.id, topic_id)
        return TopicProgressStatus(
            scroll_percent=req.scroll_percent,
            time_spent=req.time_spent,
            checkpoints_total=total_cps,
            checkpoints_confirmed=confirmed_cps,
            status="completed",
        )

    # Auto-transition to in_progress
    if progress.status == ProgressStatus.not_started and req.scroll_percent >= 25:
        progress.status = ProgressStatus.in_progress

    # Check completion conditions
    total_cps, confirmed_cps = await _get_checkpoint_counts(db, user.id, topic_id)

    can_complete = (
        req.scroll_percent >= 90
        and req.time_spent >= 30
        and confirmed_cps >= total_cps  # All checkpoints done (0/0 = True)
    )

    if can_complete and progress.status == ProgressStatus.in_progress:
        progress.status = ProgressStatus.completed

    await db.flush()

    return TopicProgressStatus(
        scroll_percent=req.scroll_percent,
        time_spent=req.time_spent,
        checkpoints_total=total_cps,
        checkpoints_confirmed=confirmed_cps,
        status=progress.status.value,
    )


async def _get_checkpoint_counts(db: AsyncSession, user_id: str, topic_id: int):
    """Return (total_checkpoints, confirmed_checkpoints) for a topic."""
    total_result = await db.execute(
        select(TopicCheckpoint).where(TopicCheckpoint.topic_id == topic_id)
    )
    total = len(total_result.scalars().all())

    confirmed_result = await db.execute(
        select(UserCheckpoint).where(
            UserCheckpoint.user_id == user_id,
            UserCheckpoint.topic_id == topic_id,
        )
    )
    confirmed = len(confirmed_result.scalars().all())

    return total, confirmed

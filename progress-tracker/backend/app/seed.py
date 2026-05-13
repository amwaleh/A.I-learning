"""
Database seeding script.
Run with:  python -m app.seed
"""

import re
import sys
import asyncio
from sqlalchemy import select, delete
from app.database import async_session, init_db
from app.models import Project, Topic, TopicCheckpoint, UserCheckpoint, UserProgress
from app.content import CURRICULUM


def extract_checkpoints(content: str | None) -> list[str]:
    """Extract checkpoint labels from <!-- checkpoint: Label --> markers in content."""
    if not content:
        return []
    return re.findall(r'<!--\s*checkpoint:\s*(.+?)\s*-->', content)


async def seed():
    await init_db()
    async with async_session() as session:
        # Check if content already exists
        existing = await session.execute(select(Project))
        if existing.scalars().first():
            print("Database already seeded – use --force to re-seed content.")
            return

        await _insert_content(session)
        await session.commit()
        print("Database seeded successfully!")


async def reseed():
    """Re-seed content tables while preserving user accounts."""
    await init_db()
    async with async_session() as session:
        # Clear content tables only (order matters for FK constraints)
        await session.execute(delete(UserCheckpoint))
        await session.execute(delete(UserProgress))
        await session.execute(delete(TopicCheckpoint))
        await session.execute(delete(Topic))
        await session.execute(delete(Project))
        await session.flush()

        await _insert_content(session)
        await session.commit()
        print("Content re-seeded successfully! User accounts preserved.")


async def _insert_content(session):
    """Insert all curriculum content and checkpoints."""
    for proj_data in CURRICULUM:
        project = Project(
            number=proj_data["number"],
            title=proj_data["title"],
            description=proj_data["description"],
        )
        session.add(project)
        await session.flush()

        for topic_data in proj_data["topics"]:
            parent_topic = Topic(
                project_id=project.id,
                title=topic_data["title"],
                content=topic_data.get("content"),
                parent_topic_id=None,
            )
            session.add(parent_topic)
            await session.flush()

            # Create checkpoints for parent topic
            for i, label in enumerate(extract_checkpoints(parent_topic.content), 1):
                session.add(TopicCheckpoint(
                    topic_id=parent_topic.id,
                    checkpoint_number=i,
                    label=label,
                ))

            for child_data in topic_data.get("children", []):
                if isinstance(child_data, str):
                    child = Topic(
                        project_id=project.id,
                        title=child_data,
                        parent_topic_id=parent_topic.id,
                    )
                else:
                    child = Topic(
                        project_id=project.id,
                        title=child_data["title"],
                        content=child_data.get("content"),
                        parent_topic_id=parent_topic.id,
                    )
                session.add(child)
                await session.flush()

                # Create checkpoints for child topic
                child_content = child_data.get("content") if isinstance(child_data, dict) else None
                for i, label in enumerate(extract_checkpoints(child_content), 1):
                    session.add(TopicCheckpoint(
                        topic_id=child.id,
                        checkpoint_number=i,
                        label=label,
                    ))

        await session.flush()


if __name__ == "__main__":
    if "--force" in sys.argv:
        asyncio.run(reseed())
    else:
        asyncio.run(seed())

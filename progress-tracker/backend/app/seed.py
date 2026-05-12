"""
Database seeding script.
Run with:  python -m app.seed
"""

import asyncio
from sqlalchemy import select
from app.database import async_session, init_db
from app.models import Project, Topic
from app.content import CURRICULUM


async def seed():
    await init_db()
    async with async_session() as session:
        # Skip if data already exists
        existing = await session.execute(select(Project))
        if existing.scalars().first():
            print("Database already seeded – skipping.")
            return

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

        await session.commit()
        print("Database seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())

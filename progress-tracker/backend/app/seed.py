"""
Database seeding script.
Run with:  python -m app.seed
"""

import asyncio
from sqlalchemy import select
from app.database import async_session, init_db
from app.models import Project, Topic


CURRICULUM: list[dict] = [
    {
        "number": 1,
        "title": "Build an LLM Playground",
        "description": "Build an LLM Playground",
        "topics": [
            {
                "title": "Pre-Training",
                "children": [
                    "Data collection",
                    "Data cleaning",
                    "Tokenization",
                    "Architecture",
                    "Text generation",
                ],
            },
            {
                "title": "Post-Training",
                "children": ["SFT", "RL and RLHF"],
            },
            {
                "title": "Evaluation",
                "children": [
                    "Traditional metrics",
                    "Task-specific benchmarks",
                    "Human evaluation",
                ],
            },
            {"title": "Chatbots' Overall Design", "children": []},
        ],
    },
    {
        "number": 2,
        "title": "Build a Customer Support Chatbot using RAGs",
        "description": "Build a Customer Support Chatbot using RAGs",
        "topics": [
            {
                "title": "Finetuning",
                "children": ["PEFT", "Adapters and LoRA"],
            },
            {
                "title": "Prompt Engineering",
                "children": [
                    "Few-shot/zero-shot",
                    "Chain-of-thought",
                    "Role-specific prompting",
                ],
            },
            {
                "title": "Retrieval",
                "children": ["Document parsing", "Indexing"],
            },
            {
                "title": "Generation",
                "children": ["Search methods", "Prompt engineering for RAGs"],
            },
            {"title": "RAFT", "children": []},
            {"title": "Evaluation", "children": []},
            {"title": "RAGs' Overall Design", "children": []},
        ],
    },
    {
        "number": 3,
        "title": "Build an Ask-the-Web Agent",
        "description": "Build an Ask-the-Web Agent",
        "topics": [
            {
                "title": "Workflows",
                "children": [
                    "Prompt chaining",
                    "Routing",
                    "Parallelization",
                    "Reflection",
                    "Orchestration-worker",
                ],
            },
            {
                "title": "Tools",
                "children": [
                    "Tool calling",
                    "Tool formatting",
                    "Tool execution",
                    "MCP",
                ],
            },
            {
                "title": "Multi-Step Agents",
                "children": ["ReACT", "Reflexion", "ReWOO", "Tree search"],
            },
            {
                "title": "Multi-Agent Systems",
                "children": ["Challenges", "Use-cases", "A2A protocol"],
            },
            {"title": "Agent Evaluation", "children": []},
        ],
    },
    {
        "number": 4,
        "title": "Build Deep Research Capability",
        "description": "Build Deep Research Capability",
        "topics": [
            {
                "title": "Inference-time Techniques",
                "children": [
                    "Inference-time scaling",
                    "CoT prompting",
                    "Parallel sampling",
                    "Sequential sampling",
                    "Tree of Thoughts",
                    "Search against verifier",
                ],
            },
            {
                "title": "Training-time Techniques",
                "children": [
                    "SFT on reasoning data",
                    "RL with verifier",
                    "Reward modeling",
                    "Self-refinement",
                    "Internalizing search",
                ],
            },
            {"title": "Local Deployment", "children": []},
        ],
    },
    {
        "number": 5,
        "title": "Build a Multi-modal Generation Agent",
        "description": "Build a Multi-modal Generation Agent",
        "topics": [
            {
                "title": "Text-to-Image",
                "children": [
                    "Data preparation",
                    "Diffusion architectures",
                    "Diffusion training",
                    "Diffusion sampling",
                    "Evaluation",
                ],
            },
            {
                "title": "Text-to-Video",
                "children": [
                    "LDM and compression",
                    "Data preparation",
                    "DiT architecture",
                    "Large-scale training",
                    "T2V overall system",
                ],
            },
        ],
    },
    {
        "number": 6,
        "title": "Ship a Portfolio-Ready AI Project",
        "description": "Capstone: Ship a Portfolio-Ready AI Project",
        "topics": [
            {"title": "Choose idea", "children": []},
            {"title": "Build implementation", "children": []},
            {"title": "Iterate with feedback", "children": []},
            {"title": "Demo presentation", "children": []},
        ],
    },
]


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
                    parent_topic_id=None,
                )
                session.add(parent_topic)
                await session.flush()

                for child_title in topic_data.get("children", []):
                    child = Topic(
                        project_id=project.id,
                        title=child_title,
                        parent_topic_id=parent_topic.id,
                    )
                    session.add(child)

            await session.flush()

        await session.commit()
        print("Database seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())

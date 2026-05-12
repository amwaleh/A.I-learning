import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import Base, get_db
from app.seed import seed_database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_seed.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestSessionLocal() as session:
        await seed_database(session)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def authenticated_client(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "seeded@example.com",
        "password": "TestPass123!",
        "display_name": "Seeded User"
    })
    response = await client.post("/auth/login", json={
        "email": "seeded@example.com",
        "password": "TestPass123!"
    })
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.mark.asyncio
async def test_seeded_projects_exist(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/projects")
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) == 6
    titles = [p["title"] for p in projects]
    assert "Build an LLM Playground" in titles


@pytest.mark.asyncio
async def test_seeded_topics_exist(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/projects/1/topics")
    assert response.status_code == 200
    topics = response.json()
    assert len(topics) > 0


@pytest.mark.asyncio
async def test_progress_update_flow(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/projects/1/topics")
    topics = response.json()
    topic_id = topics[0]["id"]

    response = await authenticated_client.patch(
        f"/progress/{topic_id}",
        json={"status": "in_progress"}
    )
    assert response.status_code == 200

    response = await authenticated_client.patch(
        f"/progress/{topic_id}",
        json={"status": "completed"}
    )
    assert response.status_code == 200

    response = await authenticated_client.get("/dashboard")
    data = response.json()
    assert data["completed_topics"] >= 1


@pytest.mark.asyncio
async def test_dashboard_with_seeded_data(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["total_topics"] > 0
    assert data["overall_percentage"] == 0.0
    assert data["streak_days"] == 0

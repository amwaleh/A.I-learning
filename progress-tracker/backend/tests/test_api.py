import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import Base, get_db
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

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
        "email": "test@example.com",
        "password": "TestPass123!",
        "display_name": "Test User"
    })
    response = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "TestPass123!"
    })
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


# ==================== Auth Tests ====================

@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "newuser@example.com",
        "password": "SecurePass123!",
        "display_name": "New User"
    })
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "newuser@example.com"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "SecurePass123!",
        "display_name": "User 1"
    })
    response = await client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "SecurePass123!",
        "display_name": "User 2"
    })
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "login@example.com",
        "password": "SecurePass123!",
        "display_name": "Login User"
    })
    response = await client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "SecurePass123!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    response = await client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "WrongPass123!"
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "refresh@example.com",
        "password": "SecurePass123!",
        "display_name": "Refresh User"
    })
    login_response = await client.post("/auth/login", json={
        "email": "refresh@example.com",
        "password": "SecurePass123!"
    })
    refresh_token = login_response.json()["refresh_token"]
    response = await client.post("/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert response.status_code == 200
    assert "access_token" in response.json()


# ==================== Projects Tests ====================

@pytest.mark.asyncio
async def test_get_projects_unauthorized(client: AsyncClient):
    response = await client.get("/projects")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_projects_authorized(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/projects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# ==================== Progress Tests ====================

@pytest.mark.asyncio
async def test_update_progress_unauthorized(client: AsyncClient):
    response = await client.patch("/progress/1", json={"status": "completed"})
    assert response.status_code == 401


# ==================== Dashboard Tests ====================

@pytest.mark.asyncio
async def test_get_dashboard_unauthorized(client: AsyncClient):
    response = await client.get("/dashboard")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_dashboard_authorized(authenticated_client: AsyncClient):
    response = await authenticated_client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total_topics" in data
    assert "overall_percentage" in data

import pytest

from datetime import datetime
from app.api.routes.auth import get_db
from app.db.models import User
from app.core.security import create_refresh_token, hash_password


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeSession:
    def __init__(self, existing_user=None):
        self.existing_user = existing_user
        self.added_user = None

    async def execute(self, query):
        return FakeResult(self.existing_user)

    def add(self, user):
        self.added_user = user

    async def commit(self):
        pass

    async def refresh(self, user):
        user.id = 1
        user.created_at = datetime.utcnow()


@pytest.mark.asyncio
async def test_register_user_success(client):
    fake_db = FakeSession(existing_user=None)

    async def override_get_db():
        yield fake_db

    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "test1234",
        },
    )

    client._transport.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_register_existing_user(client):
    existing_user = User(
        id=1,
        email="user@example.com",
        password_hash="hashed",
    )

    fake_db = FakeSession(existing_user=existing_user)

    async def override_get_db():
        yield fake_db

    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "test1234",
        },
    )

    client._transport.app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "User with this email already exists"


@pytest.mark.asyncio
async def test_login_success(client):
    existing_user = User(
        id=1,
        email="user@example.com",
        password_hash=hash_password("test1234"),
    )

    fake_db = FakeSession(existing_user=existing_user)

    async def override_get_db():
        yield fake_db

    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.post(
        "/auth/login",
        json={
            "email": "user@example.com",
            "password": "test1234",
        },
    )

    client._transport.app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    existing_user = User(
        id=1,
        email="user@example.com",
        password_hash=hash_password("correct_password"),
    )

    fake_db = FakeSession(existing_user=existing_user)

    async def override_get_db():
        yield fake_db

    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.post(
        "/auth/login",
        json={
            "email": "user@example.com",
            "password": "wrong_password",
        },
    )

    client._transport.app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


@pytest.mark.asyncio
async def test_refresh_success(client):
    refresh_token = create_refresh_token("1")

    response = await client.post(
        "/auth/refresh",
        json={
            "refresh_token": refresh_token,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_invalid_token(client):
    response = await client.post(
        "/auth/refresh",
        json={
            "refresh_token": "invalid-token",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid refresh token"
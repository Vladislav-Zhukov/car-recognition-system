import pytest

from datetime import datetime
from app.api.routes.auth import get_db
from app.db.models import User


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
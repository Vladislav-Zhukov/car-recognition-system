import pytest

import app.api.routes.prediction as prediction_module

from app.api.deps import get_current_user
from app.api.routes.prediction import get_db
from app.db.models import User

class FakeSession:
    def add(self, item):
        pass

    async def commit(self):
        pass


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_metrics(client):
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


@pytest.mark.asyncio
async def test_predict_invalid_file_extension(client):

    async def override_get_current_user():
        return User(
            id=1,
            email="user@example.com",
            password_hash="hashed",
        )

    async def override_get_db():
        yield FakeSession()

    async def fake_rate_limit(user_id):
        return None

    prediction_module.check_rate_limit = fake_rate_limit

    client._transport.app.dependency_overrides[get_current_user] = override_get_current_user
    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"not image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
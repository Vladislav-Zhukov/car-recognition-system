import pytest

import app.api.routes.prediction as prediction_module

from app.api.deps import get_current_user
from app.api.routes.prediction import get_db
from tests.mocks import FakeAsyncResult, FakeSession, fake_user


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_metrics(client):
    response = await client.get("/metrics")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_invalid_file_extension(client):
    async def override_get_current_user():
        return fake_user()

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
    assert response.json()["detail"] == "Unsupported file type. Use jpg, jpeg, png or webp."


@pytest.mark.asyncio
async def test_get_predictions_empty_history(client):
    async def override_get_current_user():
        return fake_user()

    async def override_get_db():
        yield FakeSession(result_value=[])

    client._transport.app.dependency_overrides[get_current_user] = override_get_current_user
    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.get("/predictions")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_prediction_by_id_not_found(client):
    async def override_get_current_user():
        return fake_user()

    async def override_get_db():
        yield FakeSession(result_value=None)

    client._transport.app.dependency_overrides[get_current_user] = override_get_current_user
    client._transport.app.dependency_overrides[get_db] = override_get_db

    response = await client.get("/predictions/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


@pytest.mark.asyncio
async def test_task_result_pending(client, monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "AsyncResult",
        FakeAsyncResult("PENDING").as_class(),
    )

    response = await client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_task_result_started(client, monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "AsyncResult",
        FakeAsyncResult("STARTED").as_class(),
    )

    response = await client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json()["status"] == "started"


@pytest.mark.asyncio
async def test_task_result_success(client, monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "AsyncResult",
        FakeAsyncResult(
            "SUCCESS",
            result={
                "prediction": "BMW",
                "confidence": 0.95,
            },
        ).as_class(),
    )

    response = await client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["result"]["prediction"] == "BMW"


@pytest.mark.asyncio
async def test_task_result_failure(client, monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "AsyncResult",
        FakeAsyncResult(
            "FAILURE",
            result=Exception("Model error"),
        ).as_class(),
    )

    response = await client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "Model error" in response.json()["error"]
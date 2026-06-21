import pytest

import app.api.routes.arq as arq_module
import app.api.routes.taskiq as taskiq_module


class FakeArqJob:
    job_id = "arq-job-1"


class FakeArqRedis:
    async def enqueue_job(self, *args, **kwargs):
        return FakeArqJob()


class FakeTaskiqTask:
    task_id = "taskiq-task-1"

    async def is_ready(self):
        return True

    async def wait_result(self):
        class Result:
            return_value = {
                "status": "completed",
                "predictions": [],
            }

        return Result()


@pytest.mark.asyncio
async def test_taskiq_task_unknown(client):
    response = await client.get("/taskiq/tasks/unknown-task")

    assert response.status_code == 200
    assert response.json()["status"] == "unknown"


@pytest.mark.asyncio
async def test_taskiq_task_completed(client):
    task_id = "taskiq-task-1"
    taskiq_module.taskiq_tasks_storage[task_id] = FakeTaskiqTask()

    response = await client.get(f"/taskiq/tasks/{task_id}")

    taskiq_module.taskiq_tasks_storage.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
from datetime import datetime, timezone

from app.db.models import User


def fake_user(
    user_id: int = 1,
    email: str = "user@example.com",
    password_hash: str = "hashed",
) -> User:
    return User(
        id=user_id,
        email=email,
        password_hash=password_hash,
    )


class FakeResult:
    def __init__(self, value=None):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return FakeScalars(self.value)


class FakeScalars:
    def __init__(self, value=None):
        self.value = value

    def all(self):
        return self.value or []


class FakeSession:
    def __init__(self, result_value=None):
        self.result_value = result_value
        self.added_object = None

    async def execute(self, query):
        return FakeResult(self.result_value)

    def add(self, obj):
        self.added_object = obj

    async def commit(self):
        pass

    async def refresh(self, obj):
        obj.id = 1
        obj.created_at = datetime.now(timezone.utc)


class FakeAsyncResult:
    def __init__(self, state: str, result=None):
        self.state = state
        self.result = result

    def as_class(self):
        state = self.state
        result = self.result

        class _FakeAsyncResult:
            def __init__(self, task_id, app=None):
                self.id = task_id
                self.state = state
                self.result = result

        return _FakeAsyncResult
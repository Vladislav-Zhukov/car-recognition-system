from fastapi import HTTPException, status

from app.core.redis import redis_client

REQUEST_LIMIT = 5
WINDOW_SECONDS = 60


async def check_rate_limit(user_id: int):
    key = f"rate_limit:predict:{user_id}"

    current = await redis_client.incr(key)

    if current == 1:
        await redis_client.expire(key, WINDOW_SECONDS)

    if current > REQUEST_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many prediction requests. Try again later.",
        )
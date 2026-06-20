from arq.connections import RedisSettings

from app.arq_tasks import predict_car_arq
from app.core.config import settings


def get_redis_settings() -> RedisSettings:
    redis_url = settings.REDIS_URL

    if "redis://redis:6379" in redis_url:
        return RedisSettings(host="redis", port=6379)

    return RedisSettings(host="localhost", port=6380)


class WorkerSettings:
    redis_settings = get_redis_settings()

    functions = [
        predict_car_arq,
    ]
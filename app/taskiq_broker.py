from app.core.config import settings
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

redis_url = settings.REDIS_URL

result_backend = RedisAsyncResultBackend(redis_url=redis_url)

broker = ListQueueBroker(redis_url).with_result_backend(result_backend)
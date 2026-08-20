import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

redis_client.set("test", "hello")
print(redis_client.get("test"))
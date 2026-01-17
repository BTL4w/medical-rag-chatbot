import os
import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from db.redis_cache import RedisCache


@pytest.mark.skipif(
    os.getenv("REDIS_HOST") is None,
    reason="REDIS_HOST not set",
)
def test_redis_connection():
    cache = RedisCache(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", "6379")))
    pong = cache.client.ping()
    assert pong is True

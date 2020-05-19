import os
from pathlib import Path


class Config:
    root_dir = Path(__file__).resolve().parents[1]

    device = "cpu"

    max_file_size = 16 * 1024 * 1024

    if "BATCH_SIZE" in os.environ:
        BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    else:
        BATCH_SIZE = 1

    height = 224
    width = 224
    fill_color = 0
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


class RedisConfig:
    REDIS_HOST = "redis"
    REDIS_PORT = "6379"
    REDIS_DB = 0
    REDIS_QUEUE_IMAGES = "queue"
    SERVER_SLEEP = 0.1
    CLIENT_SLEEP = 0.1
    if "FLUSH_IMG_AFTER_CLASSIFICATION" in os.environ:
        FLUSH_IMG_AFTER_CLASSIFICATION = bool(os.getenv("BATCH_SIZE"))
    else:
        FLUSH_IMG_AFTER_CLASSIFICATION = True


config = Config()
redis_config = RedisConfig()

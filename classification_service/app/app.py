from flask import Flask, request
import redis

from classification_service.config import redis_config, config
from classification_service.utils import TaskStructure


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.max_file_size
exts = {"jpg", "png", "jpeg"}

db_redis = redis.Redis(host=redis_config.REDIS_HOST,
                       port=redis_config.REDIS_PORT,
                       db=redis_config.REDIS_DB)


@app.route("/classify", methods=["POST", "GET"])
def classify():
    if request.method == "POST":
        result = classify_post(request.files['file'])
    elif request.method == "GET":
        result = classify_get(request.args.get("uuid"))
    else:
        result = {}

    return result


def classify_post(img_filestorage):
    ext = str(img_filestorage.filename.split('.')[1])
    if ext not in exts:
        result = {"status": "aborted",
                  "message": f"extension {ext} is not supported"}
        return result
    else:
        try:
            data_task = TaskStructure()
            unique_id = data_task.from_img_filestorage(img_filestorage).to_str()

            db_redis.hmset(unique_id, data_task.to_redis_hmset())
            db_redis.rpush(redis_config.REDIS_QUEUE_IMAGES, unique_id)

            result = {"status": "success",
                      "message": "added to processing",
                      "uuid": unique_id}
            return result
        except Exception:
            result = {"status": "aborted",
                      "message": f"file can't be converted to image"}
            return result


def classify_get(unique_id):
    if unique_id is None:
        result = {"status": "aborted",
                  "message": "wrong key in GET request"}
        return result

    data_task = TaskStructure()
    loaded_id = data_task.load_redis_hgetall(db_redis.hgetall(unique_id))

    if loaded_id.is_empty():
        result = {"requested uuid": unique_id,
                  "status": "aborted",
                  "message": "uuid not founded in database"}
        return result

    structure = data_task.get_structure(include_image=False)
    result = {"requested uuid": unique_id,
              "status": "success",
              "structure": structure}

    return result
